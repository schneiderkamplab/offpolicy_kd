#!/bin/bash
pip install .
mld-regmix test/mixture/test.txt --num-epochs 10 --val-every 10 --save-every 10
mld-regmix test/mixture/test.txt --num-epochs 10 --val-every 10 --save-every 10 --pretrained
mld-regmix test/mixture/test.txt --num-epochs 10 --val-every 10 --save-every 10 --distillation
mld-regmix test/mixture/test.txt --num-epochs 10 --val-every 10 --save-every 10 --pretrained --distillation
mld-standard test/gemma3/train_test1.parquet --val-data-files test/gemma3/train_test2.parquet --num-epochs 10 --val-every 10 --save-every 10
mld-standard test/gemma3/train_test1.parquet --val-data-files test/gemma3/train_test2.parquet --num-epochs 10 --val-every 10 --save-every 10 --pretrained
mld-standard test/gemma3/train_test1.parquet --val-data-files test/gemma3/train_test2.parquet --num-epochs 10 --val-every 10 --save-every 10 --distillation
mld-standard test/gemma3/train_test1.parquet --val-data-files test/gemma3/train_test2.parquet --num-epochs 10 --val-every 10 --save-every 10 --pretrained --distillation
