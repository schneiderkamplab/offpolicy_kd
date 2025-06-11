import math
import torch
from torch.nn.utils.rnn import pad_sequence

__all__ = [
    'inc_device',
    'collate_fn',
    'calculate_perplexity',
    'calculate_accuracy',
]

# setup utilities

def inc_device(device, increment):
    name, number = str(device).split(":")
    number = int(number) + increment
    device = torch.device(f"{name}:{number}")
    return device

# data utilities



# training utilities

def collate_fn(batch, max_length=4096):
    input_ids = [torch.tensor(item['input_ids'][:max_length]) for item in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = (input_ids_padded != 0).long()
    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask
    }

def calculate_perplexity(loss):
    return math.exp(loss)

def calculate_accuracy(preds, labels):
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return correct / total
