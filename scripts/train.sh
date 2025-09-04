#!/usr/bin/env bash
set -e
python -m src.train_phi15_lora \  --model_name microsoft/phi-1_5 \  --train_file data/train.jsonl \  --val_file data/val.jsonl \  --output_dir results/phi15-lora \  --max_steps 500 \  --per_device_train_batch_size 2 \  --per_device_eval_batch_size 2 \  --gradient_accumulation_steps 8 \  --lr 2e-4 --wd 0.0 --warmup_ratio 0.03 \  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \  --fp16 --use_4bit --seed 42
