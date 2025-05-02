import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, Gemma3ForCausalLM, Gemma3Config
from datasets import load_dataset

MODEL_KEY = "google/gemma-3-1b-it"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Get shared tokenizer for both teacher and student models
tokenizer = AutoTokenizer.from_pretrained(MODEL_KEY)

# Load pre-trained model
teacher_model = AutoModelForCausalLM.from_pretrained(MODEL_KEY)

# Init a fresh model
student_config = Gemma3Config.from_pretrained(MODEL_KEY)
student_model = Gemma3ForCausalLM(config=student_config)
print(student_model)


# Load the dataset
dataset = load_dataset("DDSC/europarl", split="train")


print(dataset)


loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()
distillation_objective = torch.nn.BCEWithLogitsLoss()

distillation_alpha = 1.0
step = 0
for batch in loader:
    optimizer.zero_grad()

    # Set max length to model config max length
    inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

    # Teacher model forward pass
    with torch.no_grad():
        teacher_outputs = teacher_model(**inputs)
        teacher_logits = teacher_outputs.logits

    # Student model forward pass
    student_outputs = student_model(**inputs)
    student_logits = student_outputs.logits

    # Compute the distillation loss

    # Shifting
    shifted_student_logits = student_logits[:, :-1, :].contiguous()
    shifted_teacher_logits = teacher_logits[:, :-1, :].contiguous()
    shifted_labels = inputs["input_ids"][:, 1:].contiguous()

    # LM Loss
    vocab_size = student_config.vocab_size
    # print("Shifted student logits shape", shifted_student_logits.shape)
    # print("Shifted labels shape", shifted_labels.shape)

    # Language modeling loss
    # (flatten out seq dimension)
    language_modeling_loss = criterion(shifted_student_logits.view(-1, vocab_size), shifted_labels.view(-1))

    # Distillation loss
    # (flatten out seq dimension)
    distillation_loss = distillation_objective(shifted_student_logits.view(-1, student_config.vocab_size), shifted_teacher_logits.view(-1, student_config.vocab_size))


    loss = distillation_alpha * distillation_loss + language_modeling_loss
    loss.backward()
    optimizer.step()
    step += 1
    print("Step", step, "Loss", loss)


