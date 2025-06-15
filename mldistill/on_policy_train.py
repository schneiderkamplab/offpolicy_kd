import gc
import torch
from torch.nn import functional as F

from .utils import calculate_accuracy, calculate_perplexity

__all__ = ["OnPolicyTrainer"]




class OnPolicyTrainer:
    def __init__(self, student_model, train_loader, val_loader, train_logger, val_logger, optimizer, ce_loss_fn, kl_loss_fn, teacher_model, chatdata, lmbda, beta, temperature, seq_kd,alpha=0.5, patience=10, val_steps=10):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_logger = train_logger
        self.val_logger = val_logger
        self.optimizer = optimizer
        self.ce_loss_fn = ce_loss_fn
        self.kl_loss_fn = kl_loss_fn
        self.alpha = alpha
        self.step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = patience
        self.chatdata = chatdata
        self.lmbda = lmbda
        self.beta = beta
        self.temperature = temperature
        self.seq_kd = seq_kd

    def generalized_jsd_loss(
        student_logits, teacher_logits, labels=None, beta=0.5, temperature=1.0, reduction="batchmean"
    ):
        """
        Compute the generalized Jensen-Shannon Divergence loss for knowledge distillation using F.kl_div. See Eq. (1)
        of https://huggingface.co/papers/2306.13649 for the definition.

        Args:
            student_logits: Tensor of shape (batch_size, sequence_length, vocab_size)
            teacher_logits: Tensor of shape (batch_size, sequence_length, vocab_size)
            labels: Tensor of shape (batch_size, sequence_length) with -100 for padding tokens to ignore when computing loss
            beta: Interpolation coefficient between 0 and 1 (default: 0.5)
            temperature: Softmax temperature (default: 1.0)
            reduction: Specifies the reduction to apply to the output (default: 'batchmean')

        Returns:
            loss: Scalar tensor with the generalized JSD loss
        """

        # Apply temperature scaling
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        # Compute log probabilities for student and probabilities for teacher
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if beta == 0:
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            # Compute the log of the mixture distribution
            # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
            beta = torch.tensor(beta, dtype=student_log_probs.dtype)
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_log_probs + torch.log(1 - beta), teacher_log_probs + torch.log(beta)]),
                dim=0,
            )

            # Compute KL divergences using F.kl_div
            # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is swapped compared to that defined in the paper.
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)

            # Compute the Generalized Jensen-Shannon Divergence
            jsd = beta * kl_teacher + (1 - beta) * kl_student

        # Masking
        if labels is not None:
            mask = labels != -100
            jsd = jsd[mask]

        # Apply reduction
        if reduction == "batchmean":
            return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / (jsd.size(0) * jsd.size(1))
        elif reduction == "sum":
            return jsd.sum()
        elif reduction == "mean":
            return jsd.mean()
        else:
            return jsd

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # compute student output
        outputs_student = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        # compute teacher output in eval mode
        self.teacher_model.eval()
        with torch.no_grad():
            outputs_teacher = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        # slice the logits for the generated tokens using the inputs["prompts"] lengths
        if self.chatdata:
            # gkd case
            prompt_lengths = inputs["prompts"].shape[1]
            shifted_student_logits = outputs_student.logits[:, prompt_lengths - 1: -1, :]
            shifted_teacher_logits = outputs_teacher.logits[:, prompt_lengths - 1: -1, :]
            shifted_labels = inputs["labels"][:, prompt_lengths:]
        else:
            inputs["labels"]= inputs["input_ids"]
            shifted_student_logits = outputs_student.logits[:, :-1, :]
            shifted_teacher_logits = outputs_teacher.logits[:, :-1, :]
            shifted_labels = inputs["labels"][:, 1:]




        # compute loss
        loss = self.generalized_jsd_loss(
            student_logits=shifted_student_logits,
            teacher_logits=shifted_teacher_logits,
            labels=shifted_labels,
            beta=self.beta,
        )
        # Return loss
        return (loss, outputs_student) if return_outputs else loss

    def generate_on_policy_outputs(model, inputs, generation_config, pad_token_id=None, chatdata=True):
        # Generate output with respect to the prompt only

        if chatdata:
            generated_outputs = model.generate(
                input_ids=inputs["prompts"],
                attention_mask=inputs.get("prompt_attention_mask", None),
                generation_config=generation_config,
                return_dict_in_generate=True,
            )

        else:
            generated_outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                generation_config=generation_config,
                return_dict_in_generate=True,
            )

        # Get the generated token IDs
        generated_tokens = generated_outputs.sequences
        # Calculate new attention mask
        new_attention_mask = torch.ones_like(generated_tokens)
        new_labels = generated_tokens.clone()

        # If there's pad_token_id, set attention mask to 0 for padding tokens
        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[generated_tokens == pad_token_id] = 0

        return generated_tokens, new_attention_mask, new_labels

    def train(self , num_epochs=10, alpha=0.5, val_every=100, patience=10, collect_every=None):
        if collect_every is None:
            collect_every = val_every
        self.student_model.train()
        self.teacher_model.eval()
        step=0
        for epoch in range(num_epochs):
            print(f"Starting epoch: {epoch}")
            for batch in self.train_loader:
                self.student_model.zero_grad()