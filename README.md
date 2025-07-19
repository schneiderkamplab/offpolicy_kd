# offpolicy_kd

## External Teacher Example

To run distillation with an external teacher using SensAI:

### Step 1 Start the teacher

```bash
uv run python -m sensai.logits_server \
    --model google/gemma-3-27b-pt \
    --device cuda:0 \
    --transport shared_memory \
    --num-clients 1 \
    --shm-path "/tmp/sensai_teacher_shm" \
    --num-samples 256 \
    --max-nbytes 3355443200 \
    --max-tensors 8 \
    --max-dims 8 \
    --interval 0.1 &
```

### Step 2 Run the student

```bash
mld-standard --use-external-teacher --sensai-shm-path /tmp/sensai_teacher_shm --sensai-slot-number 0 --distillation train_data.parquet --val-data-files val_data.parquet --pretrained
```

See also: [examples/external_teacher/run.bash](examples/external_teacher/run.bash)
