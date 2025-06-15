import gc
import torch
from torch.nn import functional as F

from .utils import calculate_accuracy, calculate_perplexity

__all__ = ["Trainer"]

class Trainer:

    def __init__(self, student_model, train_loader, val_loader, train_logger, val_logger, optimizer, ce_loss_fn, kl_loss_fn, teacher_model, alpha=0.5, patience=10, val_steps=10):
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
                if self.teacher_model:
                    teacher_device = self.teacher_model.device
                    teacher_logits = self.teacher_model(input_ids=input_ids.to(teacher_device), attention_mask=attention_mask.to(teacher_device)).logits
                    teacher_logits = teacher_logits[:, :-1, :].contiguous()
                    teacher_flat = teacher_logits.view(-1, teacher_logits.size(-1))
                student_logits = self.student_model(input_ids=input_ids, attention_mask=attention_mask).logits
                if self.teacher_model:
                    student_logits = student_logits.to(teacher_device)
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

    def train(self, num_epochs=10, alpha=0.5, val_every=100, patience=10, collect_every=None):
        if collect_every is None:
            collect_every = val_every
        self.student_model.train()
        self.teacher_model.eval()
        step=0
        for epoch in range(num_epochs):
            print(f"Starting epoch: {epoch}")
            for batch in self.train_loader:
                self.student_model.zero_grad()

                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]

                if self.teacher_model:
                    with torch.no_grad():
                        teacher_device = self.teacher_model.device
                        teacher_logits = self.teacher_model(input_ids=input_ids.to(teacher_device), attention_mask=attention_mask.to(teacher_device)).logits
                        teacher_logits = teacher_logits[:, :-1, :].contiguous()
                        teacher_flat = teacher_logits.view(-1, teacher_logits.size(-1))

                student_logits = self.student_model(input_ids=input_ids, attention_mask=attention_mask).logits
                if self.teacher_model:
                    student_logits = student_logits.to(self.teacher_device)
                student_logits = student_logits[:, :-1, :].contiguous()
                labels = input_ids[:, 1:].contiguous().to(teacher_device)

                student_flat = student_logits.view(-1, student_logits.size(-1))
                labels_flat = labels.view(-1)

                ce_loss = self.ce_loss_fn(student_flat, labels_flat)
                if self.teacher_model:
                    kl_loss = self.kl_loss_fn(F.log_softmax(student_flat, dim=-1), F.softmax(teacher_flat[:, :student_flat.size(dim=1)], dim=-1))
                else:
                    kl_loss = torch.tensor(0)
                loss = alpha * kl_loss + ce_loss

                loss.backward()
                self.optimizer.step()

                step += 1
                self.train_logger.log(step=step, ce_loss=ce_loss.item(), kl_loss=kl_loss.item(), total_loss=loss.item())

                if self.teacher_model:
                    del teacher_logits, teacher_flat
                del input_ids, attention_mask, student_logits
                del student_flat,labels, labels_flat
                del ce_loss, kl_loss, loss

                if step % val_every == 0:
                    val_loss, val_loss_ce, val_loss_kl, val_acc, val_ppl = self.evaluate()

                    self.val_logger.log(step=step, loss=val_loss, ce_loss=val_loss_ce, kl_loss=val_loss_kl, val_ppl=val_ppl, val_acc=val_acc)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        if self.check_pointer:
                            self.check_pointer.maybe_save(step=self.step)
                    else:
                        patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered.")
                        break
                if step % collect_every == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
            if patience_counter >= patience:
                reason = "Early stopping"
                break
        else:
            reason = f"Trained for all {num_epochs} epochs."
        print(f"Training complete: {reason}")
        if self.check_pointer:
            self.check_pointer.maybe_save(step)
