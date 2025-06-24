from accelerate import Accelerator
from functools import partial
from mltiming import timing
from pathlib import Path
import sys
import torch
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Any, Dict, List

from .train import Trainer
from .utils import *

__all__ = ["distill"]

def distill(
    times: Dict[str, Any],
    experiment: str,
    train_datasets: List[torch.utils.data.Dataset],
    val_datasets: List[torch.utils.data.Dataset],
    train_sampler: torch.utils.data.Sampler,
    val_sampler: torch.utils.data.Sampler,
    teacher: str,
    student: str,
    pretrained: bool,
    distillation: bool,
    offload_teacher: bool,
    alpha: float,
    log_every: int,
    collect_every: int,
    val_every: int,
    val_steps: int,
    save_every: int,
    save_path: str,
    save_template: str,
    log_path: str,
    run_id: str,
    num_epochs: int,
    patience: int,
    max_tokens: int | None,
    max_steps: int | None,
    max_seq_length: int,
    gradient_accumulation: int,
    batch_size: int,
    learning_rate: float,
    compile: bool,
    gradient_checkpointing: bool,
) -> None:
    with timing(times, key="timing/prepare_dataloaders"):
        accelerator = Accelerator()
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        _collate_fn = partial(collate_fn, max_seq_length=max_seq_length)
        train_combined_dataset = ConcatDataset(train_datasets)
        train_loader = DataLoader(train_combined_dataset, sampler=train_sampler, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn, num_workers=0)
        val_combined_dataset = ConcatDataset(val_datasets)
        val_loader = DataLoader(val_combined_dataset, sampler=val_sampler, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    teacher_model = None
    if distillation:
        with timing(times, key="timing/load_teacher_model"):
            teacher_config = AutoConfig.from_pretrained(teacher)
            teacher_config.max_position_embeddings = max_seq_length
            teacher_model = AutoModelForCausalLM.from_pretrained(teacher, config=teacher_config)

    with timing(times, key="timing/load_student_model"):
        student_config = AutoConfig.from_pretrained(student)
        student_config.attn_implementation = "eager"
        student_config.max_position_embeddings = max_seq_length
        if pretrained:
            student_model = AutoModelForCausalLM.from_pretrained(student, config=student_config, attn_implementation='eager')
        else:
            student_model = AutoModelForCausalLM.from_config(student_config, attn_implementation='eager')

    with timing(times, key="timing/prepare_for_training"):
        ce_loss_fn = torch.nn.CrossEntropyLoss()
        kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
        if gradient_checkpointing:
            student_model.config.use_cache = False
            student_model.gradient_checkpointing_enable()
            if teacher_model is not None:
                teacher_model.config.use_cache = False
                teacher_model.gradient_checkpointing_enable()
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)
        if offload_teacher and teacher_model:
            train_loader, val_loader, student_model, optimizer = accelerator.prepare(train_loader, val_loader, student_model, optimizer)
            teacher_model.to(inc_device(student_model.device, world_size))
        else:
            train_loader, val_loader, student_model, optimizer, teacher_model = accelerator.prepare(train_loader, val_loader, student_model, optimizer, teacher_model)
        if compile:
            student_model = torch.compile(student_model)
            if teacher_model is not None:
                teacher_model = torch.compile(teacher_model)
        if experiment is None:
            experiment = "."
        if run_id is None:
            run_id = "."
        save_path = Path(save_path) / experiment / run_id
        check_pointer = CheckPointer(student_model, save_path, save_template, save_every=save_every, disable=rank)
        log_path = Path(log_path) / experiment / run_id
        train_logger = Logger(log_path, rank)
        train_logger.append(f"train.jsonl")
        train_logger.append(sys.stdout, log_every)
        val_logger = Logger(log_path, rank)
        val_logger.append(f"val.jsonl")
        val_logger.append(sys.stdout, val_every)
        trainer = Trainer(
            student_model=student_model,
            teacher_model=teacher_model,
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
            accelerator=accelerator,
            max_tokens=max_tokens,
            max_steps=max_steps,
            gradient_accumulation=gradient_accumulation,
        )
    main_logger = Logger(None, rank, sys.stdout)
    main_logger.log(step=0, **times)

    times = {}
    with timing(times, key="timing/train"):
        trainer.train(num_epochs=num_epochs)
    times["best_val_loss"] = trainer.best_val_loss
    times["total_steps"] = trainer.step
    main_logger.log(step=0, **times)
