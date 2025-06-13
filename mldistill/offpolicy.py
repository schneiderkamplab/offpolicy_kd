from accelerate import Accelerator
from mltiming import timing
from pathlib import Path
import sys
import torch
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoModelForCausalLM, AutoConfig

from .propsampler import *
from .train import Trainer
from .utils import *

__all__ = ["distill"]

def distill(times, experiment, train_datasets, val_datasets, train_sampler, val_sampler, teacher, student, pretrained, distillation, offload_teacher, alpha, log_every, collect_every, val_every, val_steps, save_every, save_path, save_template, log_path, run_id, num_epochs, patience):
    with timing(times, key="timing/prepare_dataloaders"):
        accelerator = Accelerator()
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        train_combined_dataset = ConcatDataset(train_datasets)
        train_loader = DataLoader(train_combined_dataset, sampler=train_sampler, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)
        val_combined_dataset = ConcatDataset(val_datasets)
        val_loader = DataLoader(val_combined_dataset, sampler=val_sampler, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)
    teacher_model = None
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
            student_model = AutoModelForCausalLM.from_config(student_config, attn_implementation='eager')

    with timing(times, key="timing/prepare_for_training"):
        ce_loss_fn = torch.nn.CrossEntropyLoss()
        kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-5)
        train_loader, val_loader, student_model, optimizer = accelerator.prepare(train_loader, val_loader, student_model, optimizer)
        if teacher_model:
            teacher_model.to(inc_device(student_model.device, world_size if offload_teacher else 0))
        check_pointer = CheckPointer(student_model, save_path, save_template, save_every=save_every, rank=rank)
        log_path = Path(log_path) / experiment / run_id
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
            alpha=alpha,
            collect_every=collect_every,
            val_every=val_every,
            val_steps=val_steps,
            check_pointer=check_pointer,
            train_logger=train_logger,
            val_logger=val_logger,
            patience=patience,
        )
    main_logger = Logger(None, sys.stdout)
    main_logger.log(step=0, **times)

    times = {}
    with timing(times, key="timing/train"):
        trainer.train(num_epochs=num_epochs)
    times["best_val_loss"] = trainer.best_val_loss
    times["total_steps"] = trainer.step
    main_logger.log(step=0, **times)
