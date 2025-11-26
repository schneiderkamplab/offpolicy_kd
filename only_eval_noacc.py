import torch
import torch.nn as nn

from datasets import concatenate_datasets
from typing import IO, List, Tuple, Union
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from torch.utils.data import Sampler
from torch.nn.utils.rnn import pad_sequence
import click
from functools import partial
from transformers import AutoModelForCausalLM, AutoConfig, get_scheduler
from pathlib import Path
from torch.utils.data import DataLoader

__all__ = ["main"]
@click.command()
@click.option('--student', default="models/gemma-3-1b-pt", help="Student model identifier or path (default: models/gemma-3-1b-pt)")
@click.option('--val-data-files', multiple=True, type=click.Path(exists=True), default=None, help="Validation data files in Parquet format, if not provided it will use the same files as training data")
@click.option('--load-checkpoint', type=click.Path(exists=True), default=None, help="Path to a checkpoint to load the model from (default: None)")
@click.option('--attn-implementation', default='eager', help="Attention implementation to use (default: eager)")
@click.option('--max-seq-length', type=int, default=4096, help="Maximum sequence length for the model (default: 4096)")
@click.option('--seed', default=42, help="Random seed for data shuffling (default: 42)")
@click.option('--batch-size', default=1, type=int, help="Batch size (default: 1)")
@click.option('--tokenized', is_flag=True, type=int, help="Is the data already tokenized?")
@click.option('--pad-token-id', default=0, type=int, help="Assign the pad token id you used when tokenizing the data. Default is 0")
@click.option('--val-steps', default=None, type=int, help="Number of validation steps to run (default: no limit)")



def main(**args):
    _main(args, **args)

def _main(args, student, val_data_files, load_checkpoint, attn_implementation, max_seq_length, seed, batch_size, tokenized, pad_token_id,val_steps):
    #################
    ### LOAD DATA ###
    #################
    print("Loading datasets...")
    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, max_length=2048)
    val_datasets = load_datasets(val_data_files)
    val_combined_dataset = concatenate_datasets(val_datasets)

    if tokenized is None:

        tokenizer = AutoTokenizer.from_pretrained(student)
        if tokenizer.pad_token is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        ignore_index = tokenizer.pad_token_id
        tokenizer.padding_side = "right"
        tokenized_dataset = val_combined_dataset.map(tokenize_fn, batched=True, remove_columns=['text'])

    else:
        
        tokenizer=None
        tokenized_dataset = val_combined_dataset
        # If pre-tokenized data, use the provided padding token ID
        print("Using pre-tokenized data. Make sure it is padded from the right")
        assert pad_token_id is not None, "If tokenizer is None, padding_token_id must be provided."
        ignore_index = pad_token_id



    

    _collate_fn = partial(collate_fn, max_seq_length=max_seq_length)
    val_sampler = RandomSampler(val_datasets, seed=seed)
    val_loader = DataLoader(tokenized_dataset, sampler=val_sampler, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn, num_workers=0)
    print("Done.")

    ##################
    ### LOAD MODEL ###
    ##################
    print("Loading model from {student}")
    student_config = AutoConfig.from_pretrained(student)
    student_config.attn_implementation = attn_implementation
    student_config.max_position_embeddings = max_seq_length
    student_model = AutoModelForCausalLM.from_pretrained(student, config=student_config, attn_implementation='eager')
    
    if load_checkpoint is not None:
        state_dict = torch.load(load_checkpoint, map_location="cpu")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                new_k = k.replace("_orig_mod.", "", 1)  # strip only the first occurrence
            elif k.startswith("module."):
                new_k = k.replace("module.", "", 1)     # handle DDP case too
            else:
                new_k = k
            new_state_dict[new_k] = v
        student_model.load_state_dict(new_state_dict, strict=False)

    else:
        print("No checkpoint provided to load the model from.")

    

    ####################
    ### OTHER CONFIG ###
    ####################
    if val_steps is None or val_steps < 0:
        val_steps = None

    student_model.to("cuda" if torch.cuda.is_available() else "cpu")
    student_model.eval()

    scores = evaluate_perplexity(student_model, val_loader, tokenizer=tokenizer, pad_token_id=pad_token_id, force_max_length=max_seq_length, limit_num_steps=val_steps, ignore_index=ignore_index)
    scores["model_name"] = student
    scores["checkpoint"] = load_checkpoint
    scores["val_data_files"] = val_data_files
    scores["max_seq_length"] = max_seq_length
    scores["val_steps"] = val_steps
    print(scores) 
    # or save into some file...
    with open("eval_scores.txt", "a") as f:
        f.write(str(scores))

def evaluate_perplexity(
    model, 
    loader,
    tokenizer=None,
    force_max_length: int = None,  # All of our models should be able to handle this length
    limit_num_steps: int = None, # testing purposes
    pad_token_id: int = None,  # Only used when tokenizer is None, otherwise it is set automatically
    ignore_index: int = None,
) -> dict[str, float]:
    model.eval()
    device = model.device

    ce_loss_acc = torch.tensor(0.0, device=device)
    num_tokens_acc = torch.tensor(0, device=device)

    if limit_num_steps is not None:
        print(f"...limited to {limit_num_steps} steps")
    # Fix padding token

    ce_loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index) 

    with torch.no_grad():
        for i, batch in enumerate(loader):

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            logits = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)).logits
            # Shift by one for LM loss calc
            logits = logits[:, :-1, :].contiguous()
            labels = input_ids[:, 1:].contiguous().to(device)

            # Flatten
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)

            ce_loss_per_token = ce_loss_fn(logits_flat, labels_flat)

            num_tokens = labels_flat.ne(ignore_index).sum()  # Count non-padding tokens in labels
            
            # Accumulate values
            ce_loss_acc += ce_loss_per_token.sum()
            num_tokens_acc += num_tokens
            if limit_num_steps is not None and i + 1 >= limit_num_steps:
                break

    avg_ce_loss_per_token = ce_loss_acc / num_tokens_acc
    perplexity = torch.exp(avg_ce_loss_per_token)

    result = {
        "avg_ce_loss_per_token": avg_ce_loss_per_token.item(),
        "ppl": perplexity.item(),
        "num_tokens_seen": num_tokens_acc.item()
    }
    print(f"Validation results: {result}")
    return result

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
                        ending="parquet"

                    #elif f.endswith(".json"):
                    #    parquet_files.append(os.path.join(root, f))
                    #    ending="json"
        elif os.path.isfile(p) and (p.endswith(".parquet") ):
            parquet_files.append(p)
            ending="parquet"
        elif os.path.isfile(p) and p.endswith(".json"):
            parquet_files.append(p)
            ending="json"
        else:
            print(f"Warning: {p} is not a .parquet file or directory.")
    return parquet_files, ending

def load_datasets(
    val_data_paths: Union[str, List[str]],
):

    val_data_files, ending = find_parquet_files(val_data_paths)


    if ending=="parquet":
        val_datasets = [
        load_dataset(ending, data_files=val_data_file, split="train")
        for val_data_file in val_data_files
    ]


    else:

        val_datasets = [
            load_dataset(ending, data_files=val_data_file, split="train", **({'columns': ['input_ids']} if val_data_file.lower().endswith(".parquet") else {}))
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
    # Make sure we also use right-padding here. (it is the default but better have it fixed)
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0, padding_side="right")
    attention_mask = (input_ids_padded != 0).long()   
    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask
    }


if __name__ == "__main__":
    main()
