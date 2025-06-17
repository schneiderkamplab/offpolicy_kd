import gc
import sys
import torch
from torch.nn import functional as F
from tqdm import tqdm
from .utils_gkd import unwrap_model_for_generation
#from .train_onpolicy import train_onpolicy
from .utils import calculate_accuracy, calculate_perplexity
import random

__all__ = ["Trainer"]

class Trainer:

    def __init__(self, student_model, train_loader, val_loader, train_logger, val_logger, optimizer, ce_loss_fn, kl_loss_fn, teacher_model, check_pointer, alpha, collect_every, val_every, patience, val_steps, on_policy, lmbda, beta, seq_kd, accelerator):
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
        self.collect_every = collect_every
        self.val_every = val_every
        self.val_steps = val_steps
        self.check_pointer = check_pointer
        self.patience = patience
        self.step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.on_policy = on_policy
        self.lmbda = lmbda
        self.beta = beta
        self.seq_kd = seq_kd
        self.accelerator = accelerator

    def evaluate(self, num_steps=None):
        if num_steps is None:
            num_steps = self.val_steps
        self.student_model.eval()
        total_loss, total_ce, total_kl, total_acc = 0, 0, 0, 0
        count = 0

        with torch.no_grad():
            for i, batch in zip(range(self.val_steps), self.val_loader):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                teacher_device = self.student_model.device
                if self.teacher_model:
                    teacher_device = self.teacher_model.device
                    teacher_logits = self.teacher_model(input_ids=input_ids.to(teacher_device), attention_mask=attention_mask.to(teacher_device)).logits
                    teacher_logits = teacher_logits[:, :-1, :].contiguous()
                    teacher_flat = teacher_logits.view(-1, teacher_logits.size(-1))
                student_logits = self.student_model(input_ids=input_ids, attention_mask=attention_mask).logits.to(teacher_device)
                student_logits = student_logits[:, :-1, :].contiguous()
                labels = input_ids[:, 1:].contiguous().to(teacher_device)
                student_flat = student_logits.view(-1, student_logits.size(-1))
                labels_flat = labels.view(-1)

                ce_loss = self.ce_loss_fn(student_flat, labels_flat)
                if self.teacher_model:
                    kl_loss = self.kl_loss_fn(F.log_softmax(student_flat, dim=-1), F.softmax(teacher_flat[:, :student_flat.size(dim=1)], dim=-1))
                else:
                    kl_loss = torch.tensor(0)
                loss = self.alpha * kl_loss + ce_loss

                total_loss += loss.item()
                total_ce += ce_loss.item()
                total_kl += kl_loss.item()

                preds = torch.argmax(student_flat, dim=-1)
                acc = calculate_accuracy(preds, labels_flat)
                total_acc += acc
                count += 1

        avg_loss = total_loss / count
        avg_ce = total_ce / count
        avg_kl = total_kl / count
        avg_acc = total_acc / count
        perplexity = calculate_perplexity(avg_ce)

        return avg_loss, avg_ce, avg_kl, avg_acc, perplexity

    def train(self, num_epochs=1):
        if self.collect_every is None:
            collect_every = self.val_every
        self.student_model.train()
        if self.teacher_model:
            self.teacher_model.eval()
        val_loss, val_loss_ce, val_loss_kl, val_acc, val_ppl = self.evaluate(num_steps=self.val_steps)
        self.val_logger.log(step=self.step, loss=val_loss, ce_loss=val_loss_ce, kl_loss=val_loss_kl, val_ppl=val_ppl, val_acc=val_acc)
        for epoch in range(num_epochs):
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
                self.student_model.zero_grad()
                teacher_device = self.student_model.device

                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = input_ids[:, 1:].contiguous().to(teacher_device)

                if self.on_policy:
                    input_ids, attention_mask, labels = self.train_onpolicy(input_ids,attention_mask, labels, self.seq_kd, self.lmbda)


                if self.teacher_model:
                    with torch.no_grad():
                        teacher_device = self.teacher_model.device
                        teacher_logits = self.teacher_model(input_ids=input_ids.to(teacher_device), attention_mask=attention_mask.to(teacher_device)).logits
                        teacher_logits = teacher_logits[:, :-1, :].contiguous()
                        teacher_flat = teacher_logits.view(-1, teacher_logits.size(-1))

                student_logits = self.student_model(input_ids=input_ids, attention_mask=attention_mask).logits.to(teacher_device)
                student_logits = student_logits[:, :-1, :].contiguous()
                student_flat = student_logits.view(-1, student_logits.size(-1))
                labels_flat = labels.view(-1)

                # should we use this loss or what we were doing?
                if self.beta:
                    # compute loss
                    loss = self.generalized_jsd_loss(
                        student_logits= student_flat,
                        teacher_logits= teacher_flat[:, :student_flat.size(dim=1)],
                        labels= labels_flat,
                        beta=self.beta,
                    )

                else:
                    ce_loss = self.ce_loss_fn(student_flat, labels_flat)
                    if self.teacher_model:
                        kl_loss = self.kl_loss_fn(F.log_softmax(student_flat, dim=-1), F.softmax(teacher_flat[:, :student_flat.size(dim=1)], dim=-1))
                    else:
                        kl_loss = torch.tensor(0)
                    loss = self.alpha * kl_loss + ce_loss

                loss.backward()
                self.optimizer.step()

                self.step += 1
                self.train_logger.log(step=self.step, ce_loss=ce_loss.item(), kl_loss=kl_loss.item(), total_loss=loss.item())

                if self.teacher_model:
                    del teacher_logits, teacher_flat
                del input_ids, attention_mask, student_logits
                del student_flat,labels, labels_flat
                del ce_loss, kl_loss, loss

                if self.step % self.val_every == 0:
                    val_loss, val_loss_ce, val_loss_kl, val_acc, val_ppl = self.evaluate()
                    self.val_logger.log(step=self.step, loss=val_loss, ce_loss=val_loss_ce, kl_loss=val_loss_kl, val_ppl=val_ppl, val_acc=val_acc)

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        if self.check_pointer:
                            self.check_pointer.maybe_save(step=self.step)
                    else:
                        self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        break
                if self.step % collect_every == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
            if self.patience_counter >= self.patience:
                reason = f"Early stopping after {self.patience_counter} validations without improvement."
                break
        else:
            reason = f"Trained for all {num_epochs} epochs."
        print(f"Training complete: {reason}", file=sys.stderr)
        if self.check_pointer:
            self.check_pointer.save(self.step)
        if self.step % self.val_every != 0:
            val_loss, val_loss_ce, val_loss_kl, val_acc, val_ppl = self.evaluate(num_steps=self.val_steps)
            self.val_logger.log(step=self.step, loss=val_loss, ce_loss=val_loss_ce, kl_loss=val_loss_kl, val_ppl=val_ppl, val_acc=val_acc)

    def train_onpolicy(self, input_ids, attention_mask, labels, lmbda, seq_kd):
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        if seq_kd:
            with unwrap_model_for_generation(self.teacher_model, self.accelerator) as unwrapped_model:
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )
            input_ids = new_input_ids
            attention_mask = new_attention_mask
            labels = new_labels

        if random.random() <= lmbda:
            with unwrap_model_for_generation(self.student_model, self.accelerator) as unwrapped_model:
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )
            input_ids = new_input_ids
            attention_mask = new_attention_mask
            labels = new_labels

        return input_ids, attention_mask, labels

    @staticmethod
    def generalized_jsd_loss(self,
            student_logits, teacher_logits, labels=None, beta=0.5, temperature=1.0, reduction="batchmean"
    ):

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

    @staticmethod
    def generate_on_policy_outputs(self, model, inputs, generation_config, pad_token_id=None):
        # Generate output with respect to the prompt only

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
