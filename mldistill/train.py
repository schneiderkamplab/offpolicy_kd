from accelerate import Accelerator
import gc
import sys
import torch
from torch.nn import functional as F
from transformers import SchedulerType
from tqdm import tqdm

from .utils import *
from .utils import sparse_distillation_loss

__all__ = ["Trainer"]

class Trainer():

    def __init__(
        self,
        student_model: torch.nn.Module,
        teacher_model: torch.nn.Module | None,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        train_logger: Logger,
        val_logger: Logger,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: SchedulerType,
        ce_loss_fn: torch.nn.Module,
        kl_loss_fn: torch.nn.Module,
        alpha: float,
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
        use_external_teacher: bool = False,
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
        self.offload_optimizer = offload_optimizer
        self.use_external_teacher = use_external_teacher

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
                
                if self.use_external_teacher:
                    # Use sparse distillation loss with external teacher
                    teacher_indices = batch["teacher_logits_indices"]
                    teacher_values = batch["teacher_logits_values"]
                    student_logits = self.student_model(input_ids=input_ids, attention_mask=attention_mask).logits.to(teacher_device)
                    labels = input_ids[:, 1:].contiguous().to(teacher_device)
                    
                    loss, ce_loss, kl_loss = sparse_distillation_loss(
                        student_logits=student_logits,
                        teacher_indices=teacher_indices.to(teacher_device),
                        teacher_values=teacher_values.to(teacher_device),
                        labels=labels,
                        alpha=self.alpha
                    )
                    del teacher_indices, teacher_values
                else:
                    # Original logic for regular teacher model
                    if self.teacher_model:
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
                        del teacher_logits, teacher_flat
                    else:
                        kl_loss = torch.tensor(0)
                    loss = self.alpha * kl_loss + ce_loss
                    del student_flat, labels_flat
                
                batch_size = batch["input_ids"].size(0)
                
                losses_acc[0] += loss.detach()
                losses_acc[1] += ce_loss.detach()
                losses_acc[2] += kl_loss.detach()
                del ce_loss, kl_loss, loss

                # Calculate accuracy (need to handle both external and regular teacher cases)
                if self.use_external_teacher:
                    # For external teacher, we need to compute accuracy from student logits
                    student_logits_for_acc = self.student_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.to(teacher_device)
                    student_logits_for_acc = student_logits_for_acc[:, :-1, :].contiguous()
                    labels_for_acc = batch["input_ids"][:, 1:].contiguous().to(teacher_device)
                    student_flat_for_acc = student_logits_for_acc.view(-1, student_logits_for_acc.size(-1))
                    labels_flat_for_acc = labels_for_acc.view(-1)
                    preds = torch.argmax(student_flat_for_acc, dim=-1)
                    acc = calculate_accuracy(preds, labels_flat_for_acc)
                    del student_logits_for_acc, student_flat_for_acc, labels_for_acc, labels_flat_for_acc, preds
                else:
                    # For regular teacher, student_flat and labels_flat already exist
                    preds = torch.argmax(student_flat, dim=-1)
                    acc = calculate_accuracy(preds, labels_flat)
                    del student_flat, preds, labels_flat
                
                losses_acc[3] += acc.detach()
                del student_logits
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
                batch_size = input_ids.size(0)
                tokens += batch_size * input_ids.size(1)

                if self.use_external_teacher:
                    # Use sparse distillation loss with external teacher
                    teacher_indices = batch["teacher_logits_indices"]
                    teacher_values = batch["teacher_logits_values"]
                    student_logits = self.student_model(input_ids=input_ids, attention_mask=attention_mask).logits.to(teacher_device)
                    labels = input_ids[:, 1:].contiguous().to(teacher_device)
                    
                    loss, ce_loss, kl_loss = sparse_distillation_loss(
                        student_logits=student_logits,
                        teacher_indices=teacher_indices.to(teacher_device),
                        teacher_values=teacher_values.to(teacher_device),
                        labels=labels,
                        alpha=self.alpha
                    )
                    del teacher_indices, teacher_values, student_logits, labels
                else:
                    # Original logic for regular teacher model
                    if self.teacher_model:
                        with torch.no_grad():
                            teacher_logits = self.teacher_model(input_ids=input_ids.to(teacher_device), attention_mask=attention_mask.to(teacher_device)).logits
                            teacher_logits = teacher_logits[:, :-1, :].contiguous()
                            teacher_flat = teacher_logits.view(-1, teacher_logits.size(-1))

                    student_logits = self.student_model(input_ids=input_ids, attention_mask=attention_mask).logits.to(teacher_device)
                    student_logits = student_logits[:, :-1, :].contiguous()
                    labels = input_ids[:, 1:].contiguous().to(teacher_device)

                    student_flat = student_logits.view(-1, student_logits.size(-1))
                    labels_flat = labels.view(-1)

                    ce_loss = self.ce_loss_fn(student_flat, labels_flat)
                    del input_ids, labels, labels_flat

                    if self.teacher_model:
                        kl_loss = self.kl_loss_fn(F.log_softmax(student_flat, dim=-1), F.softmax(teacher_flat[:, :student_flat.size(dim=1)], dim=-1))
                        del teacher_logits, teacher_flat
                    else:
                        kl_loss = torch.tensor(0.0, device=teacher_device)
                    del student_logits, student_flat
                    loss = self.alpha * kl_loss + ce_loss
                
                del attention_mask
                if 'input_ids' in locals():
                    del input_ids

                self.accelerator.backward(loss)
                self.micro_step += 1
                losses[0] += loss.detach()
                losses[1] += ce_loss.detach()
                losses[2] += kl_loss.detach()
                del ce_loss, kl_loss, loss

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
