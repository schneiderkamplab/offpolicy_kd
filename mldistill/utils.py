import click
from datasets import load_dataset
from datetime import datetime
from json import dumps
import os
import gc
from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import IO, List, Tuple, Union

__all__ = [
    'CheckPointer',
    'Logger',
    'collate_fn',
    'calculate_perplexity',
    'calculate_accuracy',
    'inc_device',
    'load_datasets',
    'optimizer_to',
    'collect'
]

# setup utilities

def inc_device(
    device: torch.device,
    increment: int,
):
    name, number = str(device).split(":")
    number = int(number) + increment
    device = torch.device(f"{name}:{number}")
    return device

# data utilities

# def load_datasets(
#     train_data_files: List[torch.utils.data.Dataset],
#     val_data_files: List[torch.utils.data.Dataset],
# ):
#     train_datasets = [load_dataset("parquet", data_files=train_data_file, split="train") for train_data_file in train_data_files]
#     val_datasets = [load_dataset("parquet", data_files=val_data_file, split="train") for val_data_file in val_data_files]
#     return train_datasets, val_datasets


def load_datasets(
    train_data_paths: Union[str, List[str]],
    val_data_paths: Union[str, List[str]],
    evaluate_only: bool,
):
    train_data_files = find_parquet_files(train_data_paths)
    val_data_files = find_parquet_files(val_data_paths)

    if not train_data_files:
        raise ValueError("No valid parquet files found for training data.")
    if not val_data_files:
        raise ValueError("No valid parquet files found for validation data.")

    train_datasets = None if evaluate_only else [
        load_dataset("parquet", data_files=train_data_file, split="train", columns=['input_ids'])
        for train_data_file in train_data_files
    ]
    val_datasets = [
        load_dataset("parquet", data_files=val_data_file, split="train", columns=['input_ids'])
        for val_data_file in val_data_files
    ]

    return train_datasets, val_datasets

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


# logging utilities

class Logger():

    def __init__(
        self,
        log_path: str | None,
        disable: bool,
        overwrite: bool,
        yes: bool,
        *files: List[str | os.PathLike | IO | Tuple[str | os.PathLike | IO, int]],
    ) -> None:
        self.log_path = Path("." if log_path is None else log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.disable = disable
        self.overwrite = overwrite
        self.yes = yes
        self.files = []
        if self.disable:
            return 
        self.extend(*files)

    def extend(
        self,
        *files: List[str | os.PathLike | IO | Tuple[str | os.PathLike | IO, int]],
    ):
        if self.disable:
            return
        for file in files:
            append_args = file if isinstance(file, tuple) else (file,)
            self.append(*append_args)

    def append(
        self,
        file: str | os.PathLike | IO,
        freq: int = 1,
    ) -> None:
        if self.disable:
            return
        if isinstance(file, (str, os.PathLike)):
            path = self.log_path / file
            if path.exists():
                if not self.overwrite:
                    raise click.BadParameter(f"Output file '{path}' already exists. Use --overwrite to overwrite.")
                if not self.yes:
                    if not click.confirm(f"Output file '{path}' already exists. Do you want to delete it?"):
                        raise click.Abort()
            file = open(path, "wt")
        self.files.append((file, freq))

    def log(
        self,
        **kwargs: dict[str, str | int | float | None],
    ) -> None:
        if self.disable:
            return
        step = kwargs.pop("step", None)
        data = {
            "step": step,
            "ts": datetime.now().isoformat(),
            **kwargs,
        }
        msg = dumps(data, ensure_ascii=False)
        for file, freq in self.files:
            if step and step % freq:
                continue
            file.write(f"{msg}\n")
            file.flush()

# training utilities

class CheckPointer():

    def __init__(
        self,
        model: torch.nn.Module,
        save_path: Path,
        save_template: str,
        save_every: int = 200,
        disable: bool = False,
    ) -> None:
        self.model = model
        self.save_path = Path(save_path)
        self.save_template = save_template
        self.save_every = save_every
        self.disable = disable
        self.save_path.mkdir(parents=True, exist_ok=True)

    def maybe_save(
        self,
        step: int,
    ) -> None:
        if step % self.save_every == 0:
            self.save(step)

    def save(
        self,
        step: int,
    ) -> None:
        if not self.disable:
            checkpoint_file = self.save_path / self.save_template.format(step=step)
            torch.save(self.model.state_dict(), checkpoint_file)
            print(f"Saved checkpoint: {checkpoint_file}")

def collate_fn(
    batch: list[dict[str, torch.Tensor]],
    max_seq_length: int = 4096,
    collate_type: str = "truncate",
) -> dict[str, torch.Tensor]:
    if collate_type == "truncate":
        return collate_truncate_fn(batch, max_seq_length)
    elif collate_type == "pack":
        return collate_pack_fn(batch, max_seq_length)
    else:
        raise ValueError(f"Unknown collate type: {collate_type}")

def collate_truncate_fn(
    batch: list[dict[str, torch.Tensor]],
    max_seq_length: int = 4096,
) -> dict[str, torch.Tensor]:
    input_ids = [torch.tensor(item['input_ids'][:max_seq_length]) for item in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = (input_ids_padded != 0).long()
    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask
    }

def collate_pack_fn(batch: list[dict[str, torch.Tensor]], max_seq_length: int = 4096) -> dict[str, torch.Tensor]:
    all_ids = torch.cat([item['input_ids'] for item in batch])
    n_full, remainder_len = divmod(all_ids.size(0), max_seq_length)
    full_mask = torch.ones(max_seq_length, dtype=torch.long)
    chunks = [all_ids[i*max_seq_length:(i+1)*max_seq_length] for i in range(n_full)]
    masks = [full_mask] * n_full
    if remainder_len:
        remainder = all_ids[n_full * max_seq_length:]
        pad_len = max_seq_length - remainder_len
        chunks.append(torch.cat([remainder, torch.zeros(pad_len, dtype=all_ids.dtype)]))
        masks.append(torch.cat([torch.ones(remainder_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))
    return {
        'input_ids': torch.stack(chunks),
        'attention_mask': torch.stack(masks)
    }


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

def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        if isinstance(state, dict):
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

def collect():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()