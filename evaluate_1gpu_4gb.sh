#!/usr/bin/env bash
set -euo pipefail

# 单卡 4GB：先跑通 Industrial_and_Scientific
category="Industrial_and_Scientific"

# 这里填你的模型 ckpt 路径（方案A下载后通常是 ./Industrial_ckpt 或一个实际目录）
exp_name="./Industrial_ckpt"
exp_name_clean=$(basename "$exp_name")

echo "Category=$category | model=$exp_name_clean | SINGLE GPU 4GB"

test_file=$(ls ./data/Amazon/test/${category}*11.csv 2>/dev/null | head -1)
info_file=$(ls ./data/Amazon/info/${category}*.txt 2>/dev/null | head -1)

if [[ ! -f "$test_file" ]]; then
  echo "Error: Test file not found: $test_file"
  exit 1
fi
if [[ ! -f "$info_file" ]]; then
  echo "Error: Info file not found: $info_file"
  exit 1
fi

temp_dir="./temp/${category}-${exp_name_clean}-1gpu"
mkdir -p "$temp_dir"
cp "$test_file" "$temp_dir/0.csv"

echo "Running evaluate.py on GPU0 ..."
CUDA_VISIBLE_DEVICES=0 python -u ./evaluate.py \
  --base_model "$exp_name" \
  --info_file "$info_file" \
  --category "$category" \
  --test_data_path "$temp_dir/0.csv" \
  --result_json_data "$temp_dir/0.json" \
  --batch_size 1 \
  --num_beams 3 \
  --max_new_tokens 64 \
  --temperature 1.0 \
  --guidance_scale 1.0 \
  --length_penalty 0.0

output_dir="./results/${exp_name_clean}-1gpu"
mkdir -p "$output_dir"
cp "$temp_dir/0.json" "$output_dir/final_result_${category}.json"

echo "Calculating metrics..."
python ./calc.py \
  --path "$output_dir/final_result_${category}.json" \
  --item_path "$info_file"

echo "Done. Result: $output_dir/final_result_${category}.json"
