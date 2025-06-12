from accelerate import Accelerator
import click
from datasets import load_dataset
from mltiming import timing
from pathlib import Path
import sys
import torch
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.gemma3 import Gemma3ForCausalLM

from .propsampler import *
from .train import Trainer
from .utils import *

__all__ = ["main"]

@click.command()
@click.argument('mixture_file', type=click.Path(exists=True))
@click.option('--mixture', default=None, help="Mixture name, if not provided it will be derived from the basename of the mixture file without extension (default: None)")
@click.option('--data-dir', default=None, help="Directory containing the tokenized datasets in Parquet format, if not provided it will be derived from the parent of the mixture file (default: None)")
@click.option('--student', default="models/gemma-3-1b-pt", help="Student model identifier or path (default: models/gemma-3-1b-pt)")
@click.option('--teacher', default="models/gemma-3-4b-pt", help="Teacher model identifier or path (default: models/gemma-3-4b-pt)")
@click.option('--pretrained', is_flag=True, help="Initialize student from pretrained model instead of fresh config (default: False)")
@click.option('--distillation', is_flag=True, help="Do distillation, otherwise it will train without a teacher model (default: False)")
@click.option('--offload-teacher', is_flag=True, help="Offload teacher model to separate CPU during training (default: False)")
@click.option('--seed', default=42, help="Random seed for data shuffling (default: 42)")
@click.option('--alpha', default=1.0, type=float, help="Weight for KL divergence loss in distillation (default: 1.0)")
@click.option('--log-every', default=10, type=int, help="Log training loss every N steps (default: 10)")
@click.option('--val-every', default=100, type=int, help="Validate every N steps (default: 100)")
@click.option('--val-steps', default=10, type=int, help="Number of validation steps to run (default: 10)")
@click.option('--save-every', default=100, type=int, help="Save model checkpoint every N steps (default: 100)")
@click.option('--save-path', default="checkpoints", help="Directory to save model checkpoints (default: checkpoints)")
@click.option('--save-template', default="student_step{step}.pt", help="Template for saving model checkpoints (default: student_step{step}.pt)")
@click.option('--log-path', default="logs", help="Directory to save training logs (default: logs)")
@click.option('--run-id', default=".", help="Run ID for logging and checkpointing (default: .)")
def main(mixture_file, mixture, data_dir, student, teacher, pretrained, distillation, offload_teacher, seed, alpha, log_every, val_every, val_steps, save_every, save_path, save_template, run_id):
    times = {}
    with timing(times, key="timing/mixture_file"):
        if mixture is None:
            mixture = str(Path(mixture_file).stem)
        if data_dir:
            data_dir = Path(mixture_file).parent.parent / "gemma3"
        with open(mixture_file, "rt") as f:
            data_files = [x.strip() for x in f.readline().split(",")]
            weights = [float(x) for x in f.readline().split(",")]
        data_files, weights = zip(*((data_file, weight) for data_file, weight in zip(data_files, weights) if weight))
        train_data_files = [data_dir / f"train_{data_file}.parquet" for data_file in data_files]
        val_data_files = [data_dir / f"valid_{data_file}.parquet" for data_file in data_files]
    distill(
        times=times,
        experiment=mixture,
        train_data_files=train_data_files,
        val_data_files=val_data_files,
        weights=weights,
        teacher=teacher,
        student=student,
        pretrained=pretrained,
        distillation=distillation,
        offload_teacher=offload_teacher,
        seed=seed,
        alpha=alpha,
        log_every=log_every,
        val_every=val_every,
        val_steps=val_steps,
        save_every=save_every,
        save_path=Path(save_path),
        save_template=save_template,
        run_id=run_id,
    )

def distill(times, experiment, train_data_files, val_data_files, weights, teacher, student, pretrained, distillation, offload_teacher, seed, alpha, log_every, val_every, val_steps, save_every, save_path, save_template, run_id):
    with timing(times, key="timing/load_datasets"):
        train_datasets = [load_dataset("parquet", data_files=train_data_file, split="train") for train_data_file in train_data_files]
        val_datasets = [load_dataset("parquet", data_files=val_data_file, split="train") for val_data_file in val_data_files]
    with timing(times, key="timing/prepare_dataloaders"):
        accelerator = Accelerator()
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        train_sampler = ProportionalSampler(train_datasets, weights, seed=seed)
        train_combined_dataset = ConcatDataset(train_datasets)
        train_loader = DataLoader(train_combined_dataset, sampler=train_sampler, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)
        val_sampler = ProportionalSampler(val_datasets, weights, seed=seed)
        val_combined_dataset = ConcatDataset(val_datasets)
        val_loader = DataLoader(val_combined_dataset, sampler=val_sampler, batch_size=1, shuffle=False, collate_fn=collate_fn)
    if distillation:
        with timing(times, key="timing/load_teacher_model"):
            teacher_config = AutoConfig.from_pretrained(teacher)
            teacher_config.max_position_embeddings = 4096
            teacher_model = AutoModelForCausalLM.from_pretrained(teacher, config=teacher_config)

    with timing(times, key="timing/load_student_model"):
        student_config = AutoConfig.from_pretrained(student)
        student_config.attn_implementation = "eager"
        student_config.max_position_embeddings = 4096
        if pretrained:
            student_model = AutoModelForCausalLM.from_pretrained(student, config=student_config, attn_implementation='eager')
        else:
            student_model = Gemma3ForCausalLM(config=student_config)

    with timing(times, key="timing/prepare_for_training"):
        ce_loss_fn = torch.nn.CrossEntropyLoss()
        kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-5)
        train_loader, val_loader, student_model, optimizer = accelerator.prepare(train_loader, val_loader, student_model, optimizer)
        if teacher_model and offload_teacher:
            teacher_model.to(inc_device(student_model.device, world_size))
        check_pointer = CheckPointer(student_model, save_path, save_template, save_every=save_every, rank=rank)
        log_path = log_path / experiment / run_id
        train_logger = Logger(log_path, f"train.jsonl")
        if rank == 0:
            train_logger.append(sys.stdout, log_every)
        val_logger = Logger(log_path, f"val.jsonl")
        if rank == 0:
            val_logger.append(sys.stdout, val_every)
        trainer = Trainer(
            student_model=student_model,
            teacher_model=teacher_model if distillation else None,
            train_loader=train_loader,
            val_loader=val_loader,
            ce_loss_fn=ce_loss_fn,
            kl_loss_fn=kl_loss_fn,
            optimizer=optimizer,
            accelerator=accelerator,
            alpha=alpha,
            log_every=log_every,
            val_every=val_every,
            val_steps=val_steps,
            save_every=save_every,
            check_pointer=check_pointer,
            train_logger=train_logger,
            val_logger=val_logger
        )

    main_logger = Logger(None, sys.stdout)
    main_logger.log(step=0, **times)
    times = {}
    with timing(times, key="timing/train"):
        trainer.train(
            num_epochs=280,
            patience=10,
        )
    main_logger.log(step=0, **times)

if __name__ == "__main__":
    main()
