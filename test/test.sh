#!/bin/bash
pip install .
mld-regmix test/mixture/test.txt --num-epochs 10 --val-every 10 --save-every 100 $@
mld-regmix test/mixture/test.txt --num-epochs 10 --val-every 10 --save-every 100 --pretrained $@
mld-regmix test/mixture/test.txt --num-epochs 10 --val-every 10 --save-every 100 --distillation $@
mld-regmix test/mixture/test.txt --num-epochs 10 --val-every 10 --save-every 100 --pretrained --distillation $@
mld-standard test/gemma3/train_test1.parquet --val-data-files test/gemma3/train_test2.parquet --num-epochs 10 --val-every 10 --save-every 100 $@
mld-standard test/gemma3/train_test1.parquet --val-data-files test/gemma3/train_test2.parquet --num-epochs 10 --val-every 10 --save-every 100 --pretrained $@
mld-standard test/gemma3/train_test1.parquet --val-data-files test/gemma3/train_test2.parquet --num-epochs 10 --val-every 10 --save-every 100 --distillation $@
mld-standard test/gemma3/train_test1.parquet --val-data-files test/gemma3/train_test2.parquet --num-epochs 10 --val-every 10 --save-every 100 --pretrained --distillation $@
