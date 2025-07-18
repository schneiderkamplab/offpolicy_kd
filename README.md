# offpolicy_kd

## External Teacher Example

To run distillation with an external teacher using SensAI:

```bash
mld-standard --use-external-teacher --sensai-shm-path /tmp/sensai_teacher_shm --sensai-slot-number 0 --distillation train_data.parquet --val-data-files val_data.parquet --pretrained
```