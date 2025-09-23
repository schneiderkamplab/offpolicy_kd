#!/bin/bash
set -e

echo "Starting on-policy training..."

export NCCL_MAX_NCHANNELS=72
export NCCL_MIN_NCHANNELS=72
export JOBID=dyna_commonpile
export TORCH_COMPILE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" 
export OMP_NUM_THREADS=48
export TORCHDYNAMO_VERBOSE=1
mkdir -p logs/offpolicy/$JOBID

# Move to the directory of standard.py
cd "$(dirname "$0")/.."

# Run the script directly with accelerate
accelerate launch \
   --main_process_port 29500 \
   --multi_gpu \
   --gpu_ids 0,1,2,3,4,5,6,7 \
   --num_processes 8 \
   --num_machines 1 \
   --machine_rank 0 \
   --mixed_precision bf16 \
    -m mldistill.standard ../../data/train-dyna-common-pile-8-gemma3-chunked \
    --val-data-files ../../data/valid-dyna-giga-gemma3 \
    --max-seq-length 6144 \
    --batch-size 8 \
    --gradient-accumulation 4 \
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
    --distribution '[[1.0,0.0,0.0,0.0]]' \
    --save-path checkpoints/offpolicy \
    --patience 1000 \
    --max-new-tokens 32 \
    > >(tee logs/offpolicy/$JOBID/stdout.txt) \
    2> >(tee logs/offpolicy/$JOBID/stderr.txt >&2)
