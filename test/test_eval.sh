#!/bin/bash

pip install -e .

# Define your lists
DATASETS=("enwiki-2025_tokenizedRight.json" "github-python-2025_tokenizedRight.json")
CHECKPOINTS=("cosine_dyna_gemma3_1b_7356" "cosine_dyna_gemma3_1b_7356_fromscratch" 
"cosine_dyna_gemma3_1b_full" "cosine_dyna_gemma3_1b_full_fromscratch" 
"cosine_giga_gemma3_1b_full" "cosine_giga_gemma3_1b_full_fromscratch"
"cosine_dyna_gemma3_1b_distill_7356" "cosine_dyna_gemma3_1b_distill_7356_fromscratch" 
"cosine_dyna_gemma3_1b_distill_full" "cosine_dyna_gemma3_1b_distill_full_fromscratch" 
"cosine_giga_gemma3_1b_distill_full" "cosine_giga_gemma3_1b_distill_full_fromscratch")

# Loop over datasets
for dataset in "${DATASETS[@]}"; do
  # Loop over checkpoints
  for checkpoint in "${CHECKPOINTS[@]}"; do
    
    # Default step
    step="7356"

    # If checkpoint matches the special ones, change step
    if [[ "$checkpoint" == "cosine_dyna_gemma3_1b_distill_full" || "$checkpoint" == "cosine_dyna_gemma3_1b_distill_full_fromscratch" || "$checkpoint" == "cosine_dyna_gemma3_1b_full_fromscratch" || "$checkpoint" == "cosine_dyna_gemma3_1b_full" ]]; then
      step="12736"
    fi

    echo "Running evaluation for dataset: $dataset with checkpoint: $checkpoint and step: $step"

    python only_eval_noacc.py \
      --val-data-files "../../data/rasmus-data/$dataset" \
      --batch-size 2 \
      --student models/gemma-3-1b-pt \
      --max-seq-length 2048 \
      --val-steps -1 \
      --load-checkpoint "checkpoints/$checkpoint/student_step${step}.pt"

  done
done
