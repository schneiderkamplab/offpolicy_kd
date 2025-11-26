from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm

# === CONFIG ===
MODEL_NAME = "models/OLMo-1B"      # or local path: "models/olmo-1b"
MAX_LEN = 2048
SAVE_PATH = "data/medrag_textbooks_packed"
TEXT_COLUMN = "text"                 # change if dataset has a different field name

# === 1. Load dataset ===
print("🔹 Loading dataset...")
raw_datasets = load_dataset("MedRAG/textbooks")

# === 2. Load tokenizer ===
print("🔹 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# === 3. Tokenize ===
def tokenize_function(examples):
    return tokenizer(examples[TEXT_COLUMN])

tokenized = raw_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=8,
    remove_columns=raw_datasets["train"].column_names
)

# === 4. Pack tokens into full sequences of MAX_LEN ===
def pack_dataset(dataset, max_length):
    print("🔹 Packing dataset...")
    all_input_ids = []
    buffer = []

    for example in tqdm(dataset["input_ids"], desc="Packing"):
        buffer.extend(example)
        while len(buffer) >= max_length:
            all_input_ids.append(buffer[:max_length])
            buffer = buffer[max_length:]
    # Drop remainder (optional)
    return {"input_ids": all_input_ids}

packed_datasets = DatasetDict()
for split in tokenized.keys():
    packed = pack_dataset(tokenized[split], MAX_LEN)
    packed_datasets[split] = raw_datasets["train"].from_dict(packed)

# === 5. Save to disk ===
print(f"💾 Saving packed dataset to {SAVE_PATH}")
packed_datasets.save_to_disk(SAVE_PATH)
print("✅ Done!")
