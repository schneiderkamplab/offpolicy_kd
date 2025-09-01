#!/bin/bash
pip install -e .
export NCCL_MAX_NCHANNELS=72
export NCCL_MIN_NCHANNELS=72
export JOBID=gemma3_270m_dyna_none
mkdir -p logs/packed/$JOBID
accelerate launch \
  --multi_gpu \
  --gpu_ids all \
  --num_processes 2 \
  --num_machines 1 \
  --machine_rank 0 \
  --mixed_precision bf16 \
  -m mldistill.standard ../../data/train-dyna-gemma3-chunked \
  --val-data-files ../../data/valid-dyna-gemma3-chunked \
  --max-seq-length 1024 \
  --batch-size 16 \
  --gradient-accumulation 4 \
  --student models/gemma-3-270m-pt \
  --run-id $JOBID \
  --pretrained \
  --learning-rate 1e-5 \
  --val-every 250 \
  --save-every 250 \
  --patience 10000 \
> >(tee logs/packed/$JOBID/stdout.txt) \
2> >(tee logs/packed/$JOBID/stderr.txt >&2)
#  --teacher models/gemma-3-270m-pt \
