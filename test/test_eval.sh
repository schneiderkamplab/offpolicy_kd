#!/bin/bash

#pip install -e .

# Define your lists # 
DATASETS=("depbank" "jvj" "synne" "nordjyllandnews" "enwiki-2025_tokenizedRight.json" "github-python-2025_tokenizedRight.json")
CHECKPOINTS=("../checkpoints/packed/olmo1b_NOdistill_dyna-dolma_def/")
FOLDER="continual"
MODEL="../checkpoints/packed/olmo1b_NOdistill_dyna-dolma_def/"

echo "$(readlink -f "$MODEL")"

# Loop over datasets
for dataset in "${DATASETS[@]}"; do
  # Loop over checkpoints
  for checkpoint in "${CHECKPOINTS[@]}"; do
    
    # Default step
    echo 

    step=$(ls checkpoints/$FOLDER/$checkpoint/student_step*.pt \
       | grep -oP 'student_step\K[0-9]+' \
       | sort -n \
       | tail -1)

    echo "Using model: $MODEL"

    echo "Running evaluation for dataset: $dataset with checkpoint: $checkpoint and step: $step"

    if [[ "$dataset" == "enwiki-2025_tokenizedRight.json" || "$dataset" == "github-python-2025_tokenizedRight.json" ]]; then
      echo "using rasmus data"
      python only_eval_noacc.py \
        --val-data-files "../../data/rasmus-data/$dataset" \
        --batch-size 2 \
        --max-seq-length 1024 \
        --val-steps -1 \
        --student "$MODEL" \
        #--tokenized \
        #--load-checkpoint "checkpoints/$FOLDER/$checkpoint/student_step${step}.pt"

    else
      echo "using dyna data"
      python only_eval_noacc.py \
        --val-data-files "../../data/valid-dyna/$dataset" \
        --batch-size 2 \
        --max-seq-length 1024 \
        --val-steps -1 \
        --student "$MODEL" \
        #--tokenized \
        #--load-checkpoint "checkpoints/$FOLDER/$checkpoint/student_step${step}.pt"
    fi

  done
done

# "enwiki-2025_tokenizedRight.json" "github-python-2025_tokenizedRight.json" "depbank" "jvj" "synne" "nordjyllandnews"
# --val-data-files "../../data/rasmus-data/$dataset" \
# --val-data-files "../../data/valid-dyna-giga-gemma3/$dataset" \

# --load-checkpoint "checkpoints/$FOLDER/$checkpoint/student_step${step}.pt"
# --student "$MODEL" \