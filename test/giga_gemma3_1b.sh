#!/bin/bash
pip install -e .
# mld-regmix test/mixture/test.txt --num-epochs 10 --val-every 10 --save-every 100 $@
# mld-regmix test/mixture/test.txt --num-epochs 10 --val-every 10 --save-every 100 --pretrained $@
# mld-regmix test/mixture/test.txt --num-epochs 10 --val-every 10 --save-every 100 --distillation $@
# mld-regmix test/mixture/test.txt --num-epochs 10 --val-every 10 --save-every 100 --pretrained --distillation $@
# mld-standard test/gemma3/train_test1.parquet --val-data-files test/gemma3/train_test2.parquet --num-epochs 10 --val-every 10 --save-every 100 $@
# mld-standard test/gemma3/train_test1.parquet --val-data-files test/gemma3/train_test2.parquet --num-epochs 10 --val-every 10 --save-every 100 --pretrained $@
# mld-standard test/gemma3/train_test1.parquet --val-data-files test/gemma3/train_test2.parquet --num-epochs 10 --val-every 10 --save-every 100 --distillation $@
#-m mldistill.standard test/gemma3/train_test1.parquet --val-data-files test/gemma3/train_test2.parquet --num-epochs 10 --val-every 10 --save-every 100 --pretrained --distillation $@
export NCCL_MAX_NCHANNELS=72
export NCCL_MIN_NCHANNELS=72
export JOBID=giga_gemma3_1b
mkdir -p logs/$JOBID
accelerate launch \
  --multi_gpu \
  --gpu_ids all \
  --num_processes 2 \
  --num_machines 1 \
  --machine_rank 0 \
  --mixed_precision bf16 \
  -m mldistill.standard ../../data/train-giga-gemma3\
  --val-data-files ../../data/valid-giga-gemma3 \
  --max-seq-length 8192 \
  --batch-size 1 \
  --gradient-accumulation 32 \
  --student models/gemma-3-1b-pt \
  --run-id $JOBID \
  --pretrained \
  --teacher models/gemma-3-4b-pt \
  --learning-rate 1e-5 \
  --val-every 100 \
  --save-every 100 \
> >(tee logs/$JOBID/stdout.txt) \
2> >(tee logs/$JOBID/stderr.txt >&2)
