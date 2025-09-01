#!/bin/bash
set -e

echo "Starting on-policy training..."

export NCCL_MAX_NCHANNELS=72
export NCCL_MIN_NCHANNELS=72
export JOBID=teach_student_gen
export TORCH_COMPILE=0
mkdir -p logs/onpolicy/$JOBID

# Move to the directory of standard.py
cd "$(dirname "$0")/.."

# Run the script directly with accelerate
#accelerate launch \
#    --main_process_port 29400 \
#    --multi_gpu \
#    --gpu_ids all \
#    --num_processes 1 \
#    --num_machines 1 \
#    --machine_rank 0 \
#    --mixed_precision bf16 \
    python3 -m mldistill.standard ../../data/train-giga-gemma3 \
    --val-data-files ../../data/valid-dyna-giga-gemma3 \
    --max-seq-length 4096 \
    --batch-size 1 \
    --gradient-accumulation 2 \
    --student models/gemma-3-1b-pt \
    --run-id $JOBID \
    --pretrained \
    --distillation \
    --teacher models/gemma-3-4b-pt \
    --learning-rate 1e-5 \
    --val-every 100 \
    --save-every 100 \
    --collect-every 1 \
    --attn-implementation eager \
    --overwrite \
    --yes \
    --compile \
    --distribution '[[0.0,0.0,0.0,1.0]]' \
    --save-path checkpoints/onpolicy \
    --patience 1000 \
    --max-new-tokens 32 \
    > >(tee logs/onpolicy/$JOBID/stdout.txt) \
    2> >(tee logs/onpolicy/$JOBID/stderr.txt >&2)
