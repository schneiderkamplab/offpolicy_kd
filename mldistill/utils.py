from datasets import load_dataset
from datetime import datetime
from json import dumps
import os
from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import IO, List, Tuple

__all__ = [
    'CheckPointer',
    'Logger',
    'collate_fn',
    'calculate_perplexity',
    'calculate_accuracy',
    'inc_device',
    'load_datasets',
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

def load_datasets(
    train_data_files: List[torch.utils.data.Dataset],
    val_data_files: List[torch.utils.data.Dataset],
):
    train_datasets = [load_dataset("parquet", data_files=train_data_file, split="train") for train_data_file in train_data_files]
    val_datasets = [load_dataset("parquet", data_files=val_data_file, split="train") for val_data_file in val_data_files]
    return train_datasets, val_datasets

# logging utilities

class Logger():

    def __init__(
        self,
        log_path: str | None,
        disable: bool = False,
        *files: List[str | os.PathLike | IO | Tuple[str | os.PathLike | IO, int]],
    ) -> None:
        self.log_path = Path("." if log_path is None else log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.disable = disable
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
            file = open(self.log_path / file, "at")
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
) -> dict[str, torch.Tensor]:
    input_ids = [torch.tensor(item['input_ids'][:max_seq_length]) for item in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = (input_ids_padded != 0).long()
    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask
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
