#!/usr/bin/env bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

category="Industrial_and_Scientific"
train_file=$(ls ./data/Amazon/train/${category}*11.csv | head -1)
eval_file=$(ls ./data/Amazon/valid/${category}*11.csv | head -1)

CUDA_VISIBLE_DEVICES=0 python -u sft.py \
  --base_model="Qwen/Qwen2-1.5B" \
  --train_file="${train_file}" \
  --eval_file="${eval_file}" \
  --output_dir="output_dir/sft_embed_only_full" \
  --category="${category}" \
  --sid_index_path="./data/Amazon/index/${category}.index.json" \
  --item_meta_path="./data/Amazon/index/${category}.item.json" \
  --sample=-1 \
  --num_epochs=1 \
  --cutoff_len=256 \
  --micro_batch_size=1 \
  --batch_size=4 \
  --learning_rate=1e-4 \
  --freeze_LLM=True \
  --group_by_length=False \
  --seed=42
