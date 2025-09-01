#!/bin/bash
pip install -e .
export NCCL_MAX_NCHANNELS=72
export NCCL_MIN_NCHANNELS=72
export JOBID=gemma3_1b_pretrained_dyna5507_27b
mkdir -p logs/packed/$JOBID
accelerate launch \
  --multi_gpu \
  --gpu_ids all \
  --num_processes 4 \
  --num_machines 1 \
  --machine_rank 0 \
  --mixed_precision bf16 \
  -m mldistill.standard ../../data/train-dyna-gemma3-chunked\
  --val-data-files ../../data/valid-giga-gemma3-chunked \
  --max-seq-length 1024 \
  --batch-size 2 \
  --gradient-accumulation 32 \
  --student models/gemma-3-1b-pt \
  --run-id $JOBID \
  --distillation \
  --pretrained \
  --teacher models/gemma-3-27b-pt \
  --learning-rate 1e-5 \
  --val-every 100 \
  --val-steps 105 \
  --save-every 100 \
  --patience 1000 \
  --max-steps 5507 \
  --warmup-steps 0.05 \
  --log-path logs/packed \
  --save-path checkpoints/packed \
  --overwrite \
  --yes \
> >(tee logs/packed/$JOBID/stdout.txt) \
2> >(tee logs/packed/$JOBID/stderr.txt >&2)
