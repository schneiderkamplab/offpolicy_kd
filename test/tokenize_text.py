# from transformers import AutoTokenizer
# from datasets import load_dataset

# tokenizer = AutoTokenizer.from_pretrained(
#     "training/offpolicy_kd/models/olmo-1b",
#     trust_remote_code=True
# )

# def tokenize_function(examples):
#     return tokenizer(
#         examples["text"],
#         truncation=True,
#         padding="max_length",
#         max_length=2048,
#     )

# dataset = load_dataset(
#     "parquet",
#     data_files="data/distillOlmo/distill-dyna-dyna-0-of-1-test/distill-dyna-dyna-0-of-1-test.parquet"
# )

# tokenized_dataset = dataset.map(
#     tokenize_function,
#     batched=True,            # <— critical
#     batch_size=1000,         # adjust depending on your RAM
#     num_proc=4,              # parallelism (optional)
#     remove_columns=["text"], # saves memory
# )

# tokenized_dataset.save_to_disk(
#     "data/distillOlmo/distill-dyna-dyna-0-of-1-test_tok"
# )

# import os
# from datasets import Dataset

# arrow_dir = "data/distillOlmo/distill-dyna-dyna-0-of-1-train_tok/train"
# out_dir = "data/distillOlmo/distill-dyna-dyna-0-of-1-train_tok_parquet/train"
# os.makedirs(out_dir, exist_ok=True)

# for file in os.listdir(arrow_dir):
#     if file.endswith(".arrow"):
#         in_path = os.path.join(arrow_dir, file)
#         out_path = os.path.join(out_dir, file.replace(".arrow", ".parquet"))

#         dataset = Dataset.from_file(in_path)
#         dataset.to_parquet(out_path)

#         print("Converted:", out_path)



from transformers import AutoTokenizer
from datasets import load_dataset

# --- Load tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(
    "training/offpolicy_kd/models/olmo-1b",
    trust_remote_code=True
)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=2048,
    )

# --- Load dataset from parquet ---
dataset = load_dataset(
    "parquet",
    data_files="data/distillOlmo/distill-dyna-dolma-dyna-0-of-1-dolma-0-of-64-train.parquet"
)

# --- Tokenize ---
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    num_proc=4,
    remove_columns=["text"],
    desc="Tokenizing dataset",   # <-- this labels the HF progress bar

)

# --- Save directly to parquet ---
tokenized_dataset["train"].to_parquet(
    "data/distillOlmo/distill-dyna-dolma-dyna-0-of-1-dolma-0-of-64-train_tokenized.parquet"
)

print("Saved Parquet:", "data/distillOlmo/distill-dyna-dolma-dyna-0-of-1-dolma-0-of-64-train_tokenized.parquet")
