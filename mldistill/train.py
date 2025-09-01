from accelerate import Accelerator
import gc
import sys
import torch
from torch.nn import functional as F
from transformers import SchedulerType
from tqdm import tqdm
from .utils_gkd import unwrap_model_for_generation
import numpy as np
from .utils import *
from transformers import GenerationConfig
import random
from .gkd_config import GKDConfig
torch.autograd.set_detect_anomaly(True)

__all__ = ["Trainer"]

class Trainer():

    def __init__(
        self,
        student_model: torch.nn.Module,
        teacher_model: torch.nn.Module | None,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        train_logger: Logger,
        distillation: bool,
        val_logger: Logger,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: SchedulerType,
        ce_loss_fn: torch.nn.Module,
        kl_loss_fn: torch.nn.Module,
        alpha: float,
        beta: float,
        collect_every: int | None,
        val_every: int,
        val_steps: int,
        check_pointer: CheckPointer | None,
        patience: int,
        accelerator: Accelerator,
        max_tokens: int | None,
        max_steps: int | None,
        gradient_accumulation: int,
        offload_optimizer: bool,
        initial_step: int,
        distribution=(1, 0, 0, 0),
        max_new_tokens: int = 128,
        generation_config: GenerationConfig | None = None,
        
    ):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_logger = train_logger
        self.val_logger = val_logger
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.ce_loss_fn = ce_loss_fn
        self.kl_loss_fn = kl_loss_fn
        self.alpha = alpha
        self.beta = beta
        self.collect_every = collect_every
        self.val_every = val_every
        self.val_steps = val_steps
        self.check_pointer = check_pointer
        self.patience = patience
        self.micro_step = 0
        self.step = initial_step
        self.tokens = torch.tensor(0)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.accelerator = accelerator
        self.max_tokens = max_tokens
        self.max_steps = max_steps
        self.gradient_accumulation = gradient_accumulation
        self.distillation = distillation
        self.offload_optimizer = offload_optimizer
        self.distribution = distribution
        self.max_new_tokens = max_new_tokens
        if generation_config is None:
            self.generation_config = GenerationConfig(
            max_new_tokens= self.max_new_tokens,
          #  pad_token_id=self.processing_class.pad_token_id,
        )


    def evaluate(
        self,
        num_steps: int = None,
    ) -> dict[str, float]:
        if not len(self.val_loader):
            return {
                "loss": float('inf'),
                "ce_loss": float('inf'),
                "kl_loss": float('inf'),
                "acc": 0.0,
                "ppl": float('inf'),
            }
        if num_steps is None:
            num_steps = self.val_steps
        self.student_model.eval()
        teacher_device = self.student_model.device
        if self.teacher_model:
            self.teacher_model.eval()
            teacher_device = self.teacher_model.device
        losses_acc = torch.zeros(4, device=teacher_device)
        count = 0

        with torch.no_grad():
            for i, batch in zip(range(self.val_steps), self.val_loader):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                if self.teacher_model:
                    teacher_logits = self.teacher_model(input_ids=input_ids.to(teacher_device), attention_mask=attention_mask.to(teacher_device)).logits
                    teacher_logits = teacher_logits[:, :-1, :].contiguous()
                    teacher_flat = teacher_logits.view(-1, teacher_logits.size(-1))
                student_logits = self.student_model(input_ids=input_ids, attention_mask=attention_mask).logits.to(teacher_device)
                del attention_mask
                student_logits = student_logits[:, :-1, :].contiguous()
                labels = input_ids[:, 1:].contiguous().to(teacher_device)
                student_flat = student_logits.view(-1, student_logits.size(-1))
                labels_flat = labels.view(-1)
                batch_size = input_ids.size(0)

                ce_loss = self.ce_loss_fn(student_flat, labels_flat)
                del input_ids, labels
                if self.teacher_model:
                    kl_loss = self.kl_loss_fn(F.log_softmax(student_flat, dim=-1), F.softmax(teacher_flat[:, :student_flat.size(dim=1)], dim=-1))
                    del teacher_logits, teacher_flat
                else:
                    kl_loss = torch.tensor(0)
                del student_logits
                loss = self.alpha * kl_loss + ce_loss

                losses_acc[0] += loss.detach()
                losses_acc[1] += ce_loss.detach()
                losses_acc[2] += kl_loss.detach()
                del ce_loss, kl_loss, loss

                preds = torch.argmax(student_flat, dim=-1)
                acc = calculate_accuracy(preds, labels_flat)
                losses_acc[3] += acc.detach()
                del student_flat, preds, labels_flat
                count += batch_size

        self.accelerator.reduce(losses_acc, reduction="mean")
        losses_acc /= count
        perplexity = calculate_perplexity(losses_acc[1])

        return {
            "loss": losses_acc[0].item(),
            "ce_loss": losses_acc[1].item(),
            "kl_loss": losses_acc[2].item(),
            "acc": losses_acc[3].item(),
            "ppl": perplexity.item(),
        }
 
    def train(
        self,
        num_epochs: int = 1,
    ) -> None:
        collect_every = self.val_every if self.collect_every is None else self.collect_every
        self.student_model.train()
        self.optimizer.zero_grad()
        if self.offload_optimizer:
            optimizer_to(optimizer=self.optimizer, device='cpu')
        
        teacher_device = self.student_model.device
        if self.teacher_model:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            teacher_device = self.teacher_model.device
        self.ce_loss_fn = self.ce_loss_fn.to(teacher_device)
        self.kl_loss_fn = self.kl_loss_fn.to(teacher_device)
        self.tokens = self.tokens.to(teacher_device)
        eval_result = self.evaluate(num_steps=self.val_steps)
        self.val_logger.log(step=self.step, **eval_result)
        losses = torch.zeros(3, device=teacher_device, dtype=torch.float32)
        tokens = torch.tensor(0, device=teacher_device, dtype=torch.int64)
        progress_bar = tqdm(self.train_loader, unit="batches", total=self.max_steps)


        for epoch in range(num_epochs):
            progress_bar.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            for batch in self.train_loader:

                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]

                input_ids = input_ids.to(teacher_device)
                attention_mask = attention_mask.to(teacher_device)

                if self.beta > 0.0:
                    #print("Using causal language modeling loss")
                    student_logits = self.student_model(input_ids=input_ids, attention_mask=attention_mask).logits.to(teacher_device)
                    student_logits = student_logits[:, :-1, :].contiguous()
                    labels = input_ids[:, 1:].contiguous().to(teacher_device)
                    student_flat = student_logits.view(-1, student_logits.size(-1))
                    labels_flat = labels.view(-1)
                    ce_loss = self.ce_loss_fn(student_flat, labels_flat)
                else:
                    ce_loss = torch.tensor(0.0, device="cuda")
                #print("ce_loss:", ce_loss.item())

                self.accelerator.backward(ce_loss)

                batch_size = input_ids.size(0)
                tokens += batch_size * input_ids.size(1)

                if self.distillation:
                    new_input_ids, new_attention_mask = self.train_onpolicy(input_ids.clone().detach(), attention_mask.clone().detach(), epoch, np.array(self.distribution))
                    new_input_ids = new_input_ids.to(teacher_device)
                    new_attention_mask = new_attention_mask.to(teacher_device)

                    with torch.no_grad():
                        teacher_logits = self.teacher_model(input_ids=new_input_ids, attention_mask=new_attention_mask).logits
                        teacher_logits = teacher_logits[:, :-1, :].contiguous()
                        teacher_flat = teacher_logits.view(-1, teacher_logits.size(-1)).to(teacher_device)

                    new_student_logits = self.student_model(input_ids=new_input_ids, attention_mask=new_attention_mask).logits
                    new_student_logits = new_student_logits[:, :-1, :].contiguous()
                    new_student_flat = new_student_logits.view(-1, new_student_logits.size(-1))
                    new_student_flat = new_student_flat.to(teacher_device)
                    

                    kl_loss =  self.kl_loss_fn(F.log_softmax(new_student_flat, dim=-1), F.softmax(teacher_flat[:, :new_student_flat.size(dim=1)].detach(), dim=-1))
                else:
                    kl_loss = torch.tensor(0.0, device="cuda")

                #print("kl_loss:", kl_loss)
                
                #loss = self.alpha * kl_loss + self.beta * ce_loss

                #print("loss:", loss)
                #print("right before backward")
            
                self.accelerator.backward(kl_loss)
                self.micro_step += 1
                #losses[0] += loss.detach()
                losses[1] += ce_loss.detach()
                losses[2] += kl_loss.detach()
                

                if self.micro_step % self.gradient_accumulation == 0:
                    self.step += 1
                    progress_bar.update(1)
                    if self.offload_optimizer:
                        optimizer_to(optimizer=self.optimizer, device=self.student_model.device)
                        collect()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.lr_scheduler.step()
                    if self.offload_optimizer:
                        optimizer_to(optimizer=self.optimizer, device='cpu')
                        collect()
                    self.accelerator.reduce(losses, reduction="mean")
                    self.accelerator.reduce(tokens, reduction="sum")
                    self.tokens += tokens
                    losses /= self.gradient_accumulation * batch_size
                    self.train_logger.log(
                        step=self.step,
                        loss=losses[0].item(),
                        ce_loss=losses[1].item(),
                        kl_loss=losses[2].item(),
                        tokens=self.tokens.item(),
                    )
                    losses = torch.zeros(3, device=teacher_device, dtype=torch.float32)
                    tokens = torch.tensor(0, device=teacher_device, dtype=torch.int64)

                    if self.step % self.val_every == 0:
                        eval_result = self.evaluate(num_steps=self.val_steps)
                        self.val_logger.log(step=self.step, **eval_result)

                        if eval_result["loss"] < self.best_val_loss:
                            self.best_val_loss = eval_result["loss"]
                            self.patience_counter = 0
                            if self.check_pointer:
                                self.check_pointer.maybe_save(step=self.step)
                        else:
                            self.patience_counter += 1
                        if self.patience_counter >= self.patience:
                            break
                    if self.max_tokens and self.tokens >= self.max_tokens:
                        break
                    if self.max_steps and self.step >= self.max_steps:
                        break
                    if self.step % collect_every == 0:
                        collect()

            if self.max_tokens and self.tokens >= self.max_tokens:
                reason = f"Reached {self.tokens} tokens with a maximum of {self.max_tokens}."
                break
            if self.max_steps and self.step >= self.max_steps:
                reason = f"Reached {self.step} steps with a maximum of {self.max_steps}."
                break
            if self.patience_counter >= self.patience:
                reason = f"Early stopping after {self.patience_counter} validations without improvement."
                break
        else:
            reason = f"Trained for all {num_epochs} epochs."
        print(f"Training complete: {reason}", file=sys.stderr)
        if self.check_pointer:
            self.check_pointer.save(self.step)
        if self.step % self.val_every != 0:
            self.student_model.eval()
            eval_result = self.evaluate(num_steps=self.val_steps)
            self.val_logger.log(step=self.step, **eval_result)





    def generate_on_policy_outputs(self,model, inputs, generation_config, pad_token_id=None):
        generated_outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            generation_config=generation_config,
            return_dict_in_generate=True,
        )
        #print("size input_ids:", inputs["input_ids"].shape)

        generated_tokens = generated_outputs.sequences
        #print("generated_tokens", generated_tokens)
        #generated_tokens = generated_tokens[:, inputs["input_ids"].shape[1]:]



        #print("size generated_outputs:", generated_outputs.sequences.shape)
        new_attention_mask = torch.ones_like(generated_tokens)
        new_labels = generated_tokens.clone()

        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[generated_tokens == pad_token_id] = 0

        return generated_tokens, new_attention_mask, new_labels


    def train_onpolicy( self,
        input_ids,
        attention_mask,
        epoch,
        distribution=(1, 0, 0, 0) , 
        pad_token_id=None
    ):


        if epoch <= np.shape(distribution)[0] and np.shape(distribution)[0] > 1:
            distribution = distribution[epoch]
        else:
            distribution = distribution[0]
        #print("Using distribution:", distribution)
        #print("distribution 0 :", distribution[0])

        labels = input_ids[:, 1:].contiguous()
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        rndm = random.randint(1, 100)/100
        #print("Random number generated:", rndm)

        if rndm <= distribution[0] :
            #print("enter 0")
            new_input_ids = input_ids
            new_attention_mask = attention_mask
            new_labels = labels

        elif rndm <= distribution[0]+distribution[1]:
            #print("enter 1")
            # Mode 1: Teacher generation
            with unwrap_model_for_generation(self.teacher_model, self.accelerator) as unwrapped_model:
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, pad_token_id
                )
            
        elif rndm <= distribution[0]+distribution[1]+distribution[2]:
            #print("enter 2")
            with unwrap_model_for_generation(self.student_model, self.accelerator) as unwrapped_model:
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, pad_token_id
                )
            

        elif rndm <= distribution[0]+distribution[1]+distribution[2]+distribution[3]:
            #print("enter 3")
            
            with unwrap_model_for_generation(self.teacher_model, self.accelerator) as unwrapped_model:
                new_input_ids0, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, pad_token_id
                )

            inputs2 = {"input_ids": new_input_ids0, "attention_mask": new_attention_mask, "labels": new_labels}

            with unwrap_model_for_generation(self.student_model, self.accelerator) as unwrapped_model:
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs2, self.generation_config, pad_token_id
                )


        return new_input_ids, new_attention_mask


    def _pad_to_max_length(self,input_id_list, attention_mask_list, pad_token_id=None):
        max_len = max(t.size(1) for t in input_id_list)

        padded_inputs = []
        padded_masks = []

        for input_ids, mask in zip(input_id_list, attention_mask_list):
            pad_len = max_len - input_ids.size(1)
            if pad_len > 0:
                pad_ids = torch.full((input_ids.size(0), pad_len), pad_token_id, dtype=torch.long, device=input_ids.device)
                pad_mask = torch.zeros((mask.size(0), pad_len), dtype=torch.long, device=mask.device)
                input_ids = torch.cat([input_ids, pad_ids], dim=1)
                mask = torch.cat([mask, pad_mask], dim=1)

            padded_inputs.append(input_ids)
            padded_masks.append(mask)

        return torch.cat(padded_inputs, dim=0), torch.cat(padded_masks, dim=0)


        # if seq_kd:
        #     print("Using seq_kd")
        #     with unwrap_model_for_generation(self.teacher_model, self.accelerator) as unwrapped_model:
        #         new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
        #             unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
        #         )
        #     input_ids = new_input_ids
        #     attention_mask = new_attention_mask
        #     labels = new_labels

        # if random.random() <= lmbda:
        #     print("Using lmbda for teacher model")
        #     with unwrap_model_for_generation(self.student_model, self.accelerator) as unwrapped_model:
        #         new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
        #             unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
        #         )
        #     input_ids = new_input_ids
        #     attention_mask = new_attention_mask
        #     labels = new_labels

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

    # @staticmethod
    # def generate_on_policy_outputs(model, inputs, generation_config, pad_token_id=None):
    #     # Generate output with respect to the prompt only

    #     generated_outputs = model.generate(
    #         input_ids=inputs["input_ids"],
    #         attention_mask=inputs.get("attention_mask", None),
    #         generation_config=generation_config,
    #         return_dict_in_generate=True,
    #     )

    #     # Get the generated token IDs
    #     generated_tokens = generated_outputs.sequences
    #     # Calculate new attention mask
    #     new_attention_mask = torch.ones_like(generated_tokens)
    #     new_labels = generated_tokens.clone()

    #     # If there's pad_token_id, set attention mask to 0 for padding tokens
    #     if pad_token_id is not None:
    #         new_labels[new_labels == pad_token_id] = -100
    #         new_attention_mask[generated_tokens == pad_token_id] = 0

    #     return generated_tokens, new_attention_mask, new_labels
