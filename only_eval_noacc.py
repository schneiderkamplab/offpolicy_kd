import torch
from accelerate import Accelerator
from torch.utils.data import ConcatDataset, DataLoader
from typing import IO, List, Tuple, Union
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from torch.utils.data import Sampler
from torch.nn.utils.rnn import pad_sequence
import click
from mltiming import timing
from functools import partial
from transformers import AutoModelForCausalLM, AutoConfig, get_scheduler
from pathlib import Path

__all__ = ["main"]
@click.command()
@click.option('--student', default="models/gemma-3-1b-pt", help="Student model identifier or path (default: models/gemma-3-1b-pt)")
@click.option('--pretrained', is_flag=True, help="Initialize student from pretrained model instead of fresh config (default: False)")
@click.option('--val-data-files', multiple=True, type=click.Path(exists=True), default=None, help="Validation data files in Parquet format, if not provided it will use the same files as training data")
@click.option('--load-checkpoint', type=click.Path(exists=True), default=None, help="Path to a checkpoint to load the model from (default: None)")
@click.option('--attn-implementation', default='eager', help="Attention implementation to use (default: eager)")
@click.option('--max-seq-length', type=int, default=4096, help="Maximum sequence length for the model (default: 4096)")
@click.option('--seed', default=42, help="Random seed for data shuffling (default: 42)")
@click.option('--batch-size', default=1, type=int, help="Batch size (default: 1)")
@click.option('--val-steps', default=100, type=int, help="Number of validation steps to run (default: 100)")




def main(**args):
    _main(args, **args)

def _main(args, student, pretrained, val_data_files, load_checkpoint, attn_implementation, max_seq_length, seed, batch_size, val_steps):

    times = {}
    with timing(times, key="timing/load_datasets"):
        print("Loading datasets...")
        val_datasets = load_datasets(val_data_files)
    with timing(times, key="timing/prepare_samplers"):
        val_sampler = RandomSampler(val_datasets, seed=seed)

    _collate_fn = partial(collate_fn, max_seq_length=max_seq_length)
    val_combined_dataset = ConcatDataset(val_datasets)
    val_loader = DataLoader(val_combined_dataset, sampler=val_sampler, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn, num_workers=0)
    if val_steps < 0 :
        val_steps = len(val_loader)
    
   
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
        ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        kl_loss_fn = torch.nn.KLDivLoss(reduction="none")
        

        validator = Validator(
            student_model=student_model,
            ce_loss_fn=ce_loss_fn,
            kl_loss_fn=kl_loss_fn,
            val_loader=val_loader,
            val_steps=val_steps,
            initial_step=initial_step
        )

        validator.evaluate(num_steps=validator.val_steps)

class Validator():

    def __init__(
        self,
        student_model: torch.nn.Module,
        val_loader: torch.utils.data.DataLoader,
        ce_loss_fn: torch.nn.Module,
        kl_loss_fn: torch.nn.Module,
        val_steps: int,
        initial_step: int,
    ):
        self.student_model = student_model
        self.val_loader = val_loader
        self.ce_loss_fn = ce_loss_fn
        self.kl_loss_fn = kl_loss_fn
        self.val_steps = val_steps
        self.step = initial_step
        self.tokens = torch.tensor(0)
        self.best_val_loss = float('inf')


    def evaluate(
        self,
        num_steps: int = None,
    ) -> dict[str, float]:
        if not len(self.val_loader):
            return {
                "avg_ce_loss_per_token": float('inf'),
                "ppl": float('inf'),
            }
        print(f"Evaluating model for {num_steps} steps...")

        self.student_model.eval()
        device = self.student_model.device
        ce_loss_acc = torch.tensor(0.0, device=device)
        num_tokens_acc = torch.tensor(0, device=device)

        with torch.no_grad():
            for i, batch in zip(range(self.val_steps), self.val_loader):
                
                print(f"Validation step {i + 1}/{self.val_steps}...")
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                student_logits = self.student_model(input_ids=input_ids, attention_mask=attention_mask).logits.to(device)
                student_logits = student_logits[:, :-1, :].contiguous()
                labels = input_ids[:, 1:].contiguous().to(device)
                student_flat = student_logits.view(-1, student_logits.size(-1))
                labels_flat = labels.view(-1)

                ce_loss_per_token = self.ce_loss_fn(student_flat, labels_flat)
                ce_loss_per_token *= attention_mask[:, 1:].view(-1)  # Apply attention mask to the loss, s.th. we don't count padding tokens

                ce_loss_acc += ce_loss_per_token.sum()
                num_tokens_acc += attention_mask[:, 1:].sum()



        avg_ce_loss_per_token = ce_loss_acc / num_tokens_acc
        perplexity = calculate_perplexity(avg_ce_loss_per_token)

        result = {
            "avg_ce_loss_per_token": avg_ce_loss_per_token.item(),
            "ppl": perplexity.item(),
        }

        print(f"Validation results: {result}")
    
def calculate_perplexity(
    loss: torch.Tensor,
) -> torch.Tensor:
    return torch.exp(loss)

def calculate_accuracy(
    preds: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    correct = (preds == labels).sum()
    total = labels.numel()
    return correct / total

def find_parquet_files(paths: Union[str, List[str]]) -> List[str]:
    if isinstance(paths, str):
        paths = [paths]

    parquet_files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    if f.endswith(".parquet"):
                        parquet_files.append(os.path.join(root, f))
        elif os.path.isfile(p) and p.endswith(".parquet"):
            parquet_files.append(p)
        else:
            print(f"Warning: {p} is not a .parquet file or directory.")
    return parquet_files

def load_datasets(
    val_data_paths: Union[str, List[str]],
):
    val_data_files = find_parquet_files(val_data_paths)

    if not val_data_files:
        raise ValueError("No valid parquet files found for validation data.")

    val_datasets = [
        load_dataset("parquet", data_files=val_data_file, split="train", columns=['input_ids'])
        for val_data_file in val_data_files
    ]

    return val_datasets

class RandomSampler(Sampler):
    def __init__(
        self,
        datasets: List[torch.utils.data.Dataset],
        seed: int = 42,
        num_samples: int = None,
    ):
        self.total_length = sum(len(ds) for ds in datasets)
        self.num_samples = num_samples or self.total_length
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __iter__(self):
        perm = torch.randperm(self.total_length, generator=self.generator)[:self.num_samples]
        return iter(perm.tolist())

    def __len__(self):
        return self.num_samples
    
def collate_fn(
    batch: list[dict[str, torch.Tensor]],
    max_seq_length: int = 4096,   # All of our models should be able to handle this length
) -> dict[str, torch.Tensor]:
    input_ids = [torch.tensor(item['input_ids'][:max_seq_length]) for item in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = (input_ids_padded != 0).long()
    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask
    }


if __name__ == "__main__":
    main()