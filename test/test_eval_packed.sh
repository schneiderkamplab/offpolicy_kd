#!/bin/bash

pip install -e .

# Define your lists
DATASETS=("enwiki-2025_tokenizedRight.json" "github-python-2025_tokenizedRight.json")
CHECKPOINTS=("gemma3_1b_fromscratch_2dyna_none" "gemma3_1b_fromscratch_dyna_4b" "gemma3_1b_fromscratch_dyna_none" 
"gemma3_1b_fromscratch_dyna5507_4b" "gemma3_1b_fromscratch_dyna5507_none" "gemma3_1b_fromscratch_giga_4b" "gemma3_1b_fromscratch_giga_none" 
"gemma3_1b_pretrained_2dyna_none" "gemma3_1b_pretrained_dyna_4b" "gemma3_1b_pretrained_dyna_12b" "gemma3_1b_pretrained_dyna_none" 
"gemma3_1b_pretrained_dyna5507_4b" "gemma3_1b_pretrained_dyna5507_none" "gemma3_1b_pretrained_giga_4b" "gemma3_1b_pretrained_giga_12b" "gemma3_1b_pretrained_giga_none")

# Loop over datasets
for dataset in "${DATASETS[@]}"; do
  # Loop over checkpoints
  for checkpoint in "${CHECKPOINTS[@]}"; do

    # Find the highest step in the checkpoint folder
    step=$(ls checkpoints/packed/$checkpoint/student_step*.pt 2>/dev/null | \
           sed -E 's/.*student_step([0-9]+)\.pt/\1/' | sort -nr | head -n1)

    # Check if a valid step was found
    if [ -z "$step" ]; then
      echo "No checkpoint found for $checkpoint. Skipping..."
      continue
    fi

    echo "Running evaluation for dataset: $dataset with checkpoint: $checkpoint and step: $step"

    python only_eval_noacc.py \
      --val-data-files "../../data/rasmus-data/$dataset" \
      --batch-size 2 \
      --student models/gemma-3-1b-pt \
      --max-seq-length 2048 \
      --val-steps -1 \
      --load-checkpoint "checkpoints/packed/$checkpoint/student_step${step}.pt" \

  done
done
