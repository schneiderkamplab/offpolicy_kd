
#!/bin/bash
#pip install -e .
export NCCL_MAX_NCHANNELS=72
export NCCL_MIN_NCHANNELS=72
export JOBID=olmo1b_distill_dyna_dyna_6144

export STUDENT_MODEL="models/olmo-1b"
export TEACHER_MODEL="models/olmo-1b"

export TRAIN_DATASET="data/distillOlmo/distill-dyna-dyna-0-of-1-train_tok_parquet"
export VAL_DATASET="data/distillOlmo/distill-dyna-dyna-0-of-1-test_tok_parquet"
echo "$(realpath ../../$TRAIN_DATASET)"


export CUDA_VISIBLE_DEVICES=0

mkdir -p logs/$JOBID
mkdir -p checkpoints/packed/$JOBID

python3 -m mldistill.standard ../../$TRAIN_DATASET \
  --val-data-files ../../$VAL_DATASET \
  --max-seq-length 6144 \
  --batch-size 1 \
  --gradient-accumulation 64 \
  --student $STUDENT_MODEL \
  --run-id $JOBID \
  --distillation \
  --distribution '[[0.0,1.0,0.0,0.0]]' \
  --pretrained \
  --teacher $TEACHER_MODEL \
  --learning-rate 1e-5 \
  --val-every 100 \
  --val-steps 105 \
  --save-every 100 \
  --patience 1000 \
  --warmup-steps 0.05 \
  --log-path logs/packed \
  --save-path checkpoints/packed \
  --overwrite \
  --yes \
> >(tee logs/packed/$JOBID/stdout.txt) \
2> >(tee logs/packed/$JOBID/stderr.txt >&2)



#accelerate launch \
#  --multi_gpu \
#  --num_processes 2 \
#  --num_machines 1 \
#  --machine_rank 0 \
#  --main_process_port 29503 \
#  --mixed_precision bf16 \