
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from torch.utils.data import DataLoader

from only_eval_noacc import evaluate_perplexity

def test_Validator():
    # Load model
    path = "google/gemma-3-1b-pt"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path)

    # Create a dummy dataset
    data = {"text": ["This is a test sentence.", "Another test sentence -- this should be a very very very long sequence such that others require padding and we can check that it works well even though we need to do padding in batch mode ok?", "Yet another test sentence."]} 
    dataset = Dataset.from_dict(data)
    maxlen = 4096

    print("=== Batch size 1 ===")
    loader = DataLoader(dataset["text"], batch_size=1, shuffle=False)
    scores_bsz1 = evaluate_perplexity(model, loader, tokenizer, force_max_length=maxlen)

    print("=== Batch size 2 ===")
    loader = DataLoader(dataset["text"], batch_size=2, shuffle=False)
    scores_bsz2 = evaluate_perplexity(model, loader, tokenizer, force_max_length=maxlen)

    print("=== Batch size 3 ===")
    loader = DataLoader(dataset["text"], batch_size=3, shuffle=False)
    scores_bsz3 = evaluate_perplexity(model, loader, tokenizer, force_max_length=maxlen)

    eps = 1e-2
    assert (scores_bsz2["ppl"] - scores_bsz1["ppl"]) < eps, "Perplexity scores for batch size 2 should be close to batch size 1"
    assert (scores_bsz3["ppl"] - scores_bsz1["ppl"]) < eps, "Perplexity scores for batch size 3 should be close to batch size 1"

if __name__ == "__main__":
    test_Validator()
