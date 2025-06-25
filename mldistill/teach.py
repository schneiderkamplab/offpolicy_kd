import click
from datasets import load_dataset
from functools import partial
from mltiming import timing
import pyarrow as pa
import pyarrow.parquet as pq
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig

from .sampler import RandomSampler
from .utils import *

__all__ = ["main"]

@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path(exists=False))
@click.option('--teacher', default="models/gemma-3-4b-pt", help="Teacher model identifier or path (default: models/gemma-3-4b-pt)")
@click.option('--seed', default=42, help="Random seed for data shuffling (default: 42)")
@click.option('--max-samples', default=1_048_576, help="Maximum number of samples to run through the teacher (default: 1_048_576)")
@click.option('--max-seq-length', default=1_024, type=int, help="Maximum sequence length for training (default: 1_024)")
@click.option('--batch-size', default=1, type=int, help="Batch size (default: 1)")
@click.option('--device', default="cuda", help="Device to run the model on (default: cuda)")
def main(input_file, output_file, teacher, seed, max_samples, max_seq_length, batch_size, device):
    device = torch.device(device)
    times = {}
    with timing(times, key="timing/load_datasets"):
        dataset = load_dataset("parquet", data_files=input_file, split="train")
    with timing(times, key="timing/prepare_samplers"):
        sampler = RandomSampler([dataset], seed=seed)
    with timing(times, key="timing/prepare_dataloaders"):
        _collate_fn = partial(collate_fn, max_seq_length=max_seq_length)
        train_loader = DataLoader(dataset, sampler=None, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn, num_workers=0)
    with timing(times, key="timing/load_teacher_model"):
        teacher_config = AutoConfig.from_pretrained(teacher)
        teacher_config.max_position_embeddings = max_seq_length
        teacher_model = AutoModelForCausalLM.from_pretrained(teacher, config=teacher_config)
        teacher_model.to(device)
    main_logger = Logger(None, False, sys.stdout)
    main_logger.log(step=0, **times)
    times = {}
    samples = 0
    writer = None
    with timing(times, key="timing/teach"):
        teacher_model.eval()
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Teaching", unit="batches"):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                logits = teacher_model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)).logits
                B, T, V = logits.shape
                teacher_probs_flat = torch.softmax(logits, dim=-1).view(-1, V)
                sampled_indices_flat = torch.multinomial(teacher_probs_flat, num_samples=256, replacement=False).view(B, T, 256)
                sampled_indices = sampled_indices_flat
                log_probs = torch.log_softmax(logits, dim=-1)
                sampled_logits = torch.gather(log_probs, dim=-1, index=sampled_indices)
                samples += B
                for i in range(input_ids.size(0)):
                    sample = {
                        "input_ids": input_ids[i].tonumpy(),
                        "attention_mask": attention_mask[i].tonumpy(),
                        "teacher_indices": sampled_indices[i].tonumpy(),
                        "teacher_logits": sampled_logits[i].tonumpy(),
                    }
                    table = pa.Table.from_pydict(sample)
                    if writer is None:
                        writer = pq.ParquetWriter(output_file, table.schema, compression="zstd", compression_level=3)
                    writer.write_table(table)
                if samples >= max_samples:
                    break
    if writer is not None:
        writer.close()
    times["total_samples"] = samples
    main_logger.log(step=0, **times)

if __name__ == "__main__":
    main()
