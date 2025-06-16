import gc
import sys
import torch
from torch.nn import functional as F
from tqdm import tqdm

from .utils import calculate_accuracy, calculate_perplexity

__all__ = ["Trainer"]

class Trainer:

    def __init__(self, student_model, train_loader, val_loader, train_logger, val_logger, optimizer, ce_loss_fn, kl_loss_fn, teacher_model, check_pointer, alpha, collect_every, val_every, patience, val_steps, accelerator):
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
        self.accelerator = accelerator

    def evaluate(self, num_steps=None):
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
                teacher_device = self.student_model.device
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
                else:
                    kl_loss = torch.tensor(0)
                loss = self.alpha * kl_loss + ce_loss

                losses_acc[0] += loss
                losses_acc[1] += ce_loss
                losses_acc[2] += kl_loss

                preds = torch.argmax(student_flat, dim=-1)
                acc = calculate_accuracy(preds, labels_flat)
                losses_acc[3] += acc
                count += input_ids.size(0)

        self.accelerator.reduce(losses_acc, op=torch.distributed.ReduceOp.AVG)
        losses_acc /= count
        perplexity = calculate_perplexity(losses_acc[1].item())

        return {
            "loss": losses_acc[0].item(),
            "ce_loss": losses_acc[1].item(),
            "kl_loss": losses_acc[2].item(),
            "acc": losses_acc[3].item(),
            "ppl": perplexity
        }
 
    def train(self, num_epochs=1):
        if self.collect_every is None:
            collect_every = self.val_every
        self.student_model.train()
        teacher_device = self.student_model.device
        if self.teacher_model:
            self.teacher_model.eval()
            teacher_device = self.teacher_model.device
        eval_result = self.evaluate(num_steps=self.val_steps)
        self.val_logger.log(step=self.step, **eval_result)
        for epoch in range(num_epochs):
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
                self.student_model.zero_grad()

                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]

                if self.teacher_model:
                    with torch.no_grad():
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

                loss.backward()
                self.optimizer.step()
                self.step += 1

                losses = torch.cat([loss.unsqueeze(0), ce_loss.unsqueeze(0), kl_loss.unsqueeze(0)], dim=0)
                self.accelerator.reduce(losses, op=torch.distributed.ReduceOp.AVG)
                losses /= input_ids.size(0)
                self.train_logger.log(step=self.step, loss=losses[0].item(), ce_loss=losses[1].item(), kl_loss=losses[2].item())

                if self.teacher_model:
                    del teacher_logits, teacher_flat
                del input_ids, attention_mask, student_logits
                del student_flat,labels, labels_flat
                del ce_loss, kl_loss, loss, losses

                if self.step % self.val_every == 0:
                    eval_result = self.evaluate()
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
            eval_result = self.evaluate(num_steps=self.val_steps)
            self.val_logger.log(step=self.step, **eval_result)
