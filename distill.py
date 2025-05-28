#!/usr/bin/env python

import click
import gc
import math
import os
import torch
import torch.nn.functional as F
import json

from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,AutoConfig
)


def prepare_dataset(data_files):
    datasets = []
    for f in sorted(data_files):
        try:
            ds = load_dataset("json", data_files=f, split="train")
#            if "id" in ds.column_names and "text" in ds.column_names:
#                ds = ds.remove_columns([col for col in ds.column_names if col not in {"id", "text"}])
            if "text" in ds.column_names:
                ds = ds.remove_columns([col for col in ds.column_names if col not in {"text"}])
                datasets.append(ds)
            else:
                print(f"⚠️ Missing 'id' or 'text' in: {f}")
        except Exception as e:
            print(f"❌ Failed to load {f} due to: {e}")
    return concatenate_datasets(datasets)


def calculate_perplexity(loss):
    return math.exp(loss)


def calculate_accuracy(preds, labels):
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return correct / total


def evaluate(distillation, model, loader, tokenizer, teacher_model, device, ce_loss_fn, kl_loss_fn, alpha, name="Validation", val_steps=10):
    model.eval()
    total_loss, total_ce, total_kl, total_acc = 0, 0, 0, 0
    count = 0

    with torch.no_grad():
        for i, batch in zip(range(val_steps),loader):
            inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True, max_length=256)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            if distillation:
                teacher_logits = teacher_model(input_ids=input_ids, attention_mask=attention_mask).logits
                teacher_logits = teacher_logits[:, :-1, :].contiguous()
                teacher_flat = teacher_logits.view(-1, teacher_logits.size(-1))
            
            student_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            student_logits = student_logits[:, :-1, :].contiguous()
            labels = input_ids[:, 1:].contiguous()
            student_flat = student_logits.view(-1, student_logits.size(-1))
            labels_flat = labels.view(-1)

            ce_loss = ce_loss_fn(student_flat, labels_flat)
            if distillation:
                kl_loss = kl_loss_fn(F.log_softmax(student_flat, dim=-1), F.softmax(teacher_flat[:, :student_flat.size(dim=1)], dim=-1))
            else:
                kl_loss = torch.tensor(0)
            loss = alpha * kl_loss + ce_loss

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

    model.train()
    print(f"{name} — Loss: {avg_loss:.4f}, CE: {avg_ce:.4f}, KL: {avg_kl:.4f}, Acc: {avg_acc:.4f}, PPL: {perplexity:.4f}")
    return avg_loss, avg_ce, avg_kl, avg_acc, perplexity


@click.command()
@click.argument('data_files', nargs=-1, type=click.Path(exists=True))
@click.option('--student', default="google/gemma-3-1b-pt", help="Student model identifier")
@click.option('--teacher', default="google/gemma-3-4b-pt", help="Teacher model identifier")
@click.option('--pretrained', is_flag=True, help="Initialize student from pretrained model instead of fresh config")
@click.option('--distillation', is_flag=True, help="Do distillation, otherwise it will only run with student")


def main(data_files, teacher, student, pretrained, distillation):
    if not data_files:
        print("Please provide at least one .jsonl.gz file.")
        return

    student = student or teacher
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(teacher)
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher).to(device)

    if pretrained:
        student_model = AutoModelForCausalLM.from_pretrained(student, attn_implementation="eager").to(device)
    else:
        student_config = AutoConfig.from_pretrained(student)
        student_config.attn_implementation = "eager"
        student_model = AutoModelForCausalLM.from_config(config).to(device)

    dataset = prepare_dataset(data_files).shuffle(seed=42)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    val_test = split["test"].train_test_split(test_size=0.5)
    train_loader = torch.utils.data.DataLoader(split["train"], batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_test["train"], batch_size=1)
    test_loader = torch.utils.data.DataLoader(val_test["test"], batch_size=1)

    ce_loss_fn = torch.nn.CrossEntropyLoss()
    kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-5)

    alpha = 1.0
    val_every = 100
    val_steps = 10
    save_every = 200
    save_path = "./checkpoints"
    os.makedirs(save_path, exist_ok=True)

    best_val_loss = float("inf")
    patience = 1000
    patience_counter = 0
    step = 0
    num_epochs = 1
    

    ce_loss_history = []
    kl_loss_history = []
    total_loss_history = []

    val_loss_history = []
    val_ce_loss_history = []
    val_kl_loss_history = []
    val_ppl_history = []
    val_acc_history = []


    student_model.train()
    teacher_model.eval()
    for epoch in range(num_epochs):
        for batch in train_loader:
            student_model.zero_grad()

            inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True, max_length=256)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            if distillation:
                with torch.no_grad():
                    teacher_logits = teacher_model(input_ids=input_ids, attention_mask=attention_mask).logits
                    teacher_logits = teacher_logits[:, :-1, :].contiguous()
                    teacher_flat = teacher_logits.view(-1, teacher_logits.size(-1))

            student_logits = student_model(input_ids=input_ids, attention_mask=attention_mask).logits
            student_logits = student_logits[:, :-1, :].contiguous()
            labels = input_ids[:, 1:].contiguous()

            student_flat = student_logits.view(-1, student_logits.size(-1))
            labels_flat = labels.view(-1)

            ce_loss = ce_loss_fn(student_flat, labels_flat)
            if distillation:
                kl_loss = kl_loss_fn(F.log_softmax(student_flat, dim=-1), F.softmax(teacher_flat[:, :student_flat.size(dim=1)], dim=-1))
            else:
                kl_loss = torch.tensor(0)
            loss = alpha * kl_loss + ce_loss

            loss.backward()
            optimizer.step()

            step += 1
            if step % 10 == 0:
                
                print(f"[Step {step}] Loss: {loss.item():.4f}, CE: {ce_loss.item():.4f}, KL: {kl_loss.item():.4f}")



            ce_loss_history.append(ce_loss.item())
            kl_loss_history.append(kl_loss.item())
            total_loss_history.append(loss.item())
            with open(f"train_loss_distill_{distillation}.json", "w") as f:
                json.dump({
                    "ce_loss": ce_loss_history,
                    "kl_loss": kl_loss_history,
                    "total_loss": total_loss_history
                }, f)

            if distillation:
                del teacher_logits, teacher_flat
            del input_ids, attention_mask, student_logits
            del student_flat,labels, labels_flat
            del ce_loss, kl_loss, loss

            if step % val_every == 0:
                val_loss, val_loss_ce, val_loss_kl, val_acc, val_ppl = evaluate(distillation, student_model, val_loader, tokenizer, teacher_model, device, ce_loss_fn, kl_loss_fn, alpha, val_steps=val_steps)
                val_loss_history.append(val_loss)
                val_ce_loss_history.append(val_loss_ce)
                val_kl_loss_history.append(val_loss_kl)
                val_ppl_history.append(val_ppl)
                val_acc_history.append(val_acc)

                with open(f"val_loss_distill_{distillation}.json", "w") as f:
                    json.dump({
                        "loss": val_loss_history,
                        "ce_loss": val_ce_loss_history,
                        "kl_loss": val_kl_loss_history,
                        "val_ppl": val_ppl_history,
                        "val_acc": val_acc_history
                    }, f)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if step % save_every == 0:
                        checkpoint_file = os.path.join(save_path, f"student_step{step}.pt")
                        torch.save(student_model.state_dict(), checkpoint_file)
                        print(f"✅ Saved checkpoint: {checkpoint_file}")
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print("⏹️ Early stopping triggered.")
                    break
            #print("Clearing memory ...")
            gc.collect()
            if torch.cuda.is_available():
                #print("Clearing CUDA cache ...")
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                #print("Clearing MPS cache ...")
                torch.mps.empty_cache()
        if patience_counter >= patience:
            break

    

    

    print("🔍 Final Evaluation on Test Set...")

    test_loss, test_acc, test_ppl = evaluate(distillation, student_model, test_loader, tokenizer, teacher_model, device,ce_loss_fn, kl_loss_fn, alpha, name="Test")
    with open("test_loss.json", "w") as f:
        json.dump({
            "loss": test_loss,
            "test_acc": test_acc,
            "test_ppl": test_ppl
        }, f)

if __name__ == "__main__":
    main()
