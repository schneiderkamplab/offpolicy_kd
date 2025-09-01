from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from functools import partial
from mltiming import timing
from pathlib import Path
import sys
import torch
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoModelForCausalLM, AutoConfig, get_scheduler
from typing import Any, Dict, List
import numpy as np
from .train import Trainer
from .utils import *

__all__ = ["distill"]

def distill(
    args: Dict[str, Any],
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
    beta: float,
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
    warmup_steps: float | None,
    max_seq_length: int,
    gradient_accumulation: int,
    batch_size: int,
    learning_rate: float,
    compile: bool,
    gradient_checkpointing: bool,
    offload_optimizer: bool,
    overwrite: bool,
    yes: bool,
    attn_implementation: str,
    lr_scheduler_type: str,
    evaluate_only: bool,
    load_checkpoint: str | None,
    collate_type: str,
    distribution: tuple[float, float, float, float],
    max_new_tokens: int
) -> None:
    with timing(times, key="timing/prepare_dataloaders"):
        # accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
        accelerator = Accelerator()
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        print(f"Rank {rank} using device {accelerator.device}")
        _collate_fn = partial(collate_fn, max_seq_length=max_seq_length, collate_type=collate_type)

        if all(row[0] < 1 for row in distribution): # in this case, we are doing on policy distillation (if the first num<1, the other % of that comes from on-policy), so we need to adjust the max_seq_length, we'll need to adjust this when we do offpolicy in one epoch and then switch to on-policy
            _collate_fn = partial(collate_fn, max_seq_length=max_seq_length-max_new_tokens, collate_type=collate_type)
        train_combined_dataset = None if evaluate_only else ConcatDataset(train_datasets)
        train_loader = None if evaluate_only else DataLoader(train_combined_dataset, sampler=train_sampler, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn, num_workers=0)
        val_combined_dataset = ConcatDataset(val_datasets)
        val_loader = DataLoader(val_combined_dataset, sampler=val_sampler, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    teacher_model = None
    if distillation:
        with timing(times, key="timing/load_teacher_model"):
            teacher_config = AutoConfig.from_pretrained(teacher)
            teacher_config.attn_implementation = attn_implementation
            teacher_config.max_position_embeddings = max_seq_length
            teacher_model = AutoModelForCausalLM.from_pretrained(teacher, config=teacher_config)

    with timing(times, key="timing/load_student_model"):
        student_config = AutoConfig.from_pretrained(student)
        student_config.attn_implementation = attn_implementation
        student_config.max_position_embeddings = max_seq_length
        if pretrained:
            student_model = AutoModelForCausalLM.from_pretrained(student, config=student_config, attn_implementation='eager')
        else:
            student_model = AutoModelForCausalLM.from_config(student_config, attn_implementation='eager')
        if load_checkpoint is None:
            initial_step = 0
        else:
            initial_step = int(Path(load_checkpoint).stem.split("step")[-1])
            state_dict = torch.load(load_checkpoint, map_location="cpu")
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
            student_model.load_state_dict(state_dict)
    if initial_step:
        with timing(times, key="timing/fast_forward_val_loader"):
            skip_vals = initial_step // val_steps # skip initial evaluation, too
            for _ in range(skip_vals):
                for _ in zip(range(val_steps), val_loader):
                    pass

    with timing(times, key="timing/prepare_for_training"):
        ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
        if evaluate_only:
            if offload_teacher and teacher_model:
                val_loader, student_model = accelerator.prepare(val_loader, student_model)
                teacher_model.to(inc_device(student_model.device, world_size))
            else:
                val_loader, student_model, teacher_model = accelerator.prepare(val_loader, student_model, teacher_model)
            optimizer = None
            lr_scheduler = None
            check_pointer = None
            train_logger = None
            val_logger = None
        else:
            if gradient_checkpointing:
                student_model.config.use_cache = False
                student_model.gradient_checkpointing_enable()
            optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)
            if max_steps is None:
                max_steps = len(train_loader) // gradient_accumulation // world_size * num_epochs
            lr_scheduler = get_scheduler(
                name=lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=0 if warmup_steps is None else int(max_steps * warmup_steps),
                num_training_steps=max_steps,
            )
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
            train_logger = Logger(log_path, rank, overwrite, yes)
            train_logger.append(f"train.jsonl")
            train_logger.append(sys.stdout, log_every)
            val_logger = Logger(log_path, rank, overwrite, yes)
            val_logger.append(f"val.jsonl")
            val_logger.append(sys.stdout, val_every)
        trainer = Trainer(
            student_model=student_model,
            teacher_model=teacher_model,
            train_loader=train_loader,
            distillation=distillation,
            val_loader=val_loader,
            ce_loss_fn=ce_loss_fn,
            kl_loss_fn=kl_loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler, 
            alpha=alpha,
            beta=beta,
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
            offload_optimizer=offload_optimizer,
            initial_step=initial_step,
            distribution=distribution,
            max_new_tokens=max_new_tokens,
        )
    main_logger = Logger(None, rank, overwrite, yes, sys.stderr if evaluate_only else sys.stdout)
    main_logger.log(step=0, **args)
    main_logger.log(step=0, **times)

    times = {}
    if evaluate_only:
        try:
            eval_result = trainer.evaluate(num_steps=trainer.val_steps)
        finally:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        eval_logger = Logger(None, rank, overwrite, yes, sys.stdout)
        eval_logger.log(step=initial_step, **eval_result)
    else:
        with timing(times, key="timing/train"):
            try:
                trainer.train(num_epochs=num_epochs)
            finally:
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
        times["best_val_loss"] = trainer.best_val_loss
        times["total_steps"] = trainer.step
    main_logger.log(step=0, **times)