#!/bin/bash

#pip install -e .

# Define your lists # 
#DATASETS=("nordjyllandnews")
DATASETS=("depbank" "jvj" "synne" "enwiki-2025_tokenizedRight.json" "github-python-2025_tokenizedRight.json" "nordjyllandnews")
#CHECKPOINTS=("olmo1b_NOdistill_dyna_dyna") #
CHECKPOINTS=("dyna_commonpile") # "olmo1b_distill1b_dyna-dyna" "olmo1b_NOdistill_dyna_dyna_afterstop" "olmo1b_NOdistill_dyna-cp_def"
FOLDER="offpolicy" #
MODEL="models/gemma-3-1b-pt" # THIS IS GEMMA CAREFUL!

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
        --student "$MODEL" \
        --tokenized \
        --load-checkpoint "checkpoints/$FOLDER/$checkpoint/student_step15700.pt"
        # --val-steps 128 \

    else
      echo "using dyna data"
      python only_eval_noacc.py \
        --val-data-files "../../data/valid-dyna/$dataset" \
        --batch-size 2 \
        --max-seq-length 1024 \
        --student "$MODEL" \
        --load-checkpoint "checkpoints/$FOLDER/$checkpoint/student_step15700.pt"
        #--tokenized \
        #
        # --val-steps 128 \
    fi

  done
done

# "enwiki-2025_tokenizedRight.json" "github-python-2025_tokenizedRight.json" "depbank" "jvj" "synne" "nordjyllandnews"
# --val-data-files "../../data/rasmus-data/$dataset" \
# --val-data-files "../../data/valid-dyna-giga-gemma3/$dataset" \

# --load-checkpoint "checkpoints/$FOLDER/$checkpoint/student_step${step}.pt"
# --student "$MODEL" \
