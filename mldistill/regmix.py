import os
from mltiming import timing
import sys

from .propsampler import ProportionalSampler, IndexedMultiDataset
from .utils import collate_fn, inc_device, calculate_accuracy, calculate_perplexity

with timing(message=f"Importing with {sys.executable} in {os.getcwd()}"):
    from accelerate import Accelerator
    import click
    from datasets import load_dataset
    from datetime import datetime
    import gc
    import json
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoConfig
    from transformers.models.gemma3 import Gemma3ForCausalLM

def evaluate(distillation, model, loader, teacher_model, ce_loss_fn, kl_loss_fn, alpha, name="Validation", val_steps=10):
    model.eval()
    total_loss, total_ce, total_kl, total_acc = 0, 0, 0, 0
    count = 0

    with torch.no_grad():
        for i, batch in zip(range(val_steps),loader):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            if distillation:
                teacher_logits = teacher_model(input_ids=input_ids.to(teacher_model.device), attention_mask=attention_mask.to(teacher_model.device)).logits
                teacher_logits = teacher_logits[:, :-1, :].contiguous()
                teacher_flat = teacher_logits.view(-1, teacher_logits.size(-1))
            student_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.to(teacher_model.device)
            student_logits = student_logits[:, :-1, :].contiguous()
            labels = input_ids[:, 1:].contiguous().to(teacher_model.device)
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
    ts = datetime.now().isoformat()
    print(f"{ts}: [{name}] Loss: {avg_loss:.4f}, CE: {avg_ce:.4f}, KL: {avg_kl:.4f}, Acc: {avg_acc:.4f}, PPL: {perplexity:.4f}")
    return avg_loss, avg_ce, avg_kl, avg_acc, perplexity

@click.command()
@click.argument('data_files', nargs=-1, type=click.Path(exists=True))
@click.option('--student', default="models/gemma-3-1b-pt", help="Student model identifier")
@click.option('--teacher', default="models/gemma-3-1b-pt", help="Teacher model identifier")
@click.option('--pretrained', is_flag=True, help="Initialize student from pretrained model instead of fresh config")
@click.option('--distillation', is_flag=True, help="Do distillation, otherwise it will only run with student")
@click.option('--seed', default=42, help="Random see (default: 42)")
def main(data_files, teacher, student, pretrained, distillation, seed):
    if len(data_files) != 1 or not data_files[0].endswith(".txt"):
        raise ValueError("Expecting one mixture file with .txt suffix!")
    mixture = data_files[0].split(".txt")[0].split("/")[-1]
    data_dir = "/".join(data_files[0].split("/")[:-2])
    data_dir = os.path.join(data_dir, "gemma3")
    _lines = open(data_files[0], "rt").readlines()
    data_files = [x.strip() for x in _lines[0].split(",")]
    weights = [float(x) for x in _lines[1].split(",")]
    data_files, weights = zip(*((data_file, weight) for data_file, weight in zip(data_files, weights) if weight))
    print(f"Found {len(weights)} mixtures from {data_dir}")

    with timing(message="Loading train datasets"):
        train_datasets = [
            load_dataset(
                "parquet",
                data_files=os.path.join(
                    data_dir,
                    f"train_{os.path.splitext(os.path.basename(data_file))[0]}.parquet"
                ),
                split="train",
            )
            for data_file in data_files
        ]

    with timing(message="Loading validation datasets"):
        val_datasets = [
            load_dataset(
                "parquet",
                data_files=os.path.join(
                    data_dir,
                    f"valid_{os.path.splitext(os.path.basename(data_file))[0]}.parquet"
                ),
                split="train",
            )
            for data_file in data_files
        ]

    with timing(message="Preparing weighter data samplers"):
        accelerator = Accelerator()
        rank = accelerator.process_index
        world_size = accelerator.num_processes

        train_sampler = ProportionalSampler(train_datasets, weights, seed=seed, rank=rank, world_size=world_size)
        train_combined_dataset = IndexedMultiDataset(train_datasets, train_sampler.index_mapping)
        train_loader = DataLoader(train_combined_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

        val_sampler = ProportionalSampler(val_datasets, weights, seed=seed, rank=rank, world_size=world_size)
        val_combined_dataset = IndexedMultiDataset(val_datasets, val_sampler.index_mapping)
        val_loader = DataLoader(val_combined_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    if distillation:
        with timing(message="Loading teacher model"):
            teacher_config = AutoConfig.from_pretrained(teacher)
            teacher_config.max_position_embeddings = 4096
            teacher_model = AutoModelForCausalLM.from_pretrained(teacher, config=teacher_config)

    with timing(message="Loading student model"):
        student_config = AutoConfig.from_pretrained(student)
        student_config.attn_implementation = "eager"
        student_config.max_position_embeddings = 4096
        if pretrained:
            student_model = AutoModelForCausalLM.from_pretrained(student, config=student_config, attn_implementation='eager')
        else:
            student_model = Gemma3ForCausalLM(config=student_config)

    with timing(message="Preparing for training"):
        ce_loss_fn = torch.nn.CrossEntropyLoss()
        kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-5)
        train_loader, val_loader, student_model, optimizer = accelerator.prepare(train_loader, val_loader, student_model, optimizer)
        if distillation:
            teacher_model.to(inc_device(student_model.device, world_size))

    jobid = os.environ.get("JOBID", "interactive")

    alpha = 1.0
    log_every = 10
    val_every = 100
    val_steps = 10
    save_every = 100
    save_path = f"./checkpoints/{mixture}/{jobid}"
    os.makedirs(save_path, exist_ok=True)

    best_val_loss = float("inf")
    patience = 1000
    patience_counter = 0
    step = 0
    num_epochs = 1

    log_path = f"./logs/{mixture}/{jobid}"
    os.makedirs(log_path, exist_ok=True)
    train_log = os.path.join(log_path, "train_loss_distill_{distillation}_{pretrained}.jsonl")
    val_log = os.path.join(log_path,f"val_loss_distill_{distillation}_{pretrained}.jsonl")

    student_model.train()
    teacher_model.eval()
    for epoch in range(num_epochs):
        print(f"epoch: {epoch}")
        for batch in train_loader:
            student_model.zero_grad()

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            if distillation:
                with torch.no_grad():
                    teacher_logits = teacher_model(input_ids=input_ids.to(teacher_model.device), attention_mask=attention_mask.to(teacher_model.device)).logits
                    teacher_logits = teacher_logits[:, :-1, :].contiguous()
                    teacher_flat = teacher_logits.view(-1, teacher_logits.size(-1))

            student_logits = student_model(input_ids=input_ids, attention_mask=attention_mask).logits.to(teacher_model.device)
            student_logits = student_logits[:, :-1, :].contiguous()
            labels = input_ids[:, 1:].contiguous().to(teacher_model.device)

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
            if step % log_every == 0:
                ts = datetime.now().isoformat()
                print(f"{ts}: [Step {step}] Loss: {loss.item():.4f}, CE: {ce_loss.item():.4f}, KL: {kl_loss.item():.4f}")

            with open(train_log, "at" if step > 1 else "wt") as f:
                ts = datetime.now().isoformat()
                json.dump({
                    "step": step,
                    "ce_loss": ce_loss.item(),
                    "kl_loss": kl_loss.item(),
                    "total_loss": loss.item(),
                    "ts": ts,
                }, f)
                f.write("\n")

            if distillation:
                del teacher_logits, teacher_flat
            del input_ids, attention_mask, student_logits
            del student_flat,labels, labels_flat
            del ce_loss, kl_loss, loss

            if step % val_every == 0:
                val_loss, val_loss_ce, val_loss_kl, val_acc, val_ppl = evaluate(distillation, student_model, val_loader, teacher_model, ce_loss_fn, kl_loss_fn, alpha, val_steps=val_steps)

                with open(val_log, "at" if step > 1 else "wt") as f:
                    ts = datetime.now().isoformat()
                    json.dump({
                        "step": step,
                        "loss": val_loss,
                        "ce_loss": val_loss_ce,
                        "kl_loss": val_loss_kl,
                        "val_ppl": val_ppl,
                        "val_acc": val_acc,
                        "ts": ts,
                    }, f)
                    f.write("\n")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if rank == 0 and step % save_every == 0:
                        checkpoint_file = os.path.join(save_path, f"{mixture}_student_step{step}.pt")
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

if __name__ == "__main__":
    main()
