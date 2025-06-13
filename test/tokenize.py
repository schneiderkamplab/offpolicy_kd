from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("../models/gemma-3-1b-pt")

def process(sample):
    sample["input_ids"] = tokenizer(sample["text"], return_tensors="np")["input_ids"][0]
    return sample
