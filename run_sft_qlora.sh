#!/usr/bin/env bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

# Auto-configure proxy when running inside WSL and proxy env is not set.
# Override default port by setting WSL_PROXY_PORT, e.g. `export WSL_PROXY_PORT=7897`.
if grep -qi microsoft /proc/version 2>/dev/null; then
  if [[ -z "${http_proxy:-}" && -z "${HTTP_PROXY:-}" && -z "${https_proxy:-}" && -z "${HTTPS_PROXY:-}" ]]; then
    proxy_host=$(ip route 2>/dev/null | awk '/^default/ {print $3; exit}')
    proxy_port="${WSL_PROXY_PORT:-7890}"
    if [[ -n "${proxy_host}" ]]; then
      export http_proxy="http://${proxy_host}:${proxy_port}"
      export https_proxy="http://${proxy_host}:${proxy_port}"
      export HTTP_PROXY="${http_proxy}"
      export HTTPS_PROXY="${https_proxy}"
      export no_proxy="localhost,127.0.0.1,::1"
      export NO_PROXY="${no_proxy}"
      echo "[run_sft_qlora] WSL proxy enabled: ${http_proxy}"
    fi
  fi
fi

category="Industrial_and_Scientific"
train_file=$(ls ./data/Amazon/train/${category}*11.csv | head -1)
eval_file=$(ls ./data/Amazon/valid/${category}*11.csv | head -1)

CUDA_VISIBLE_DEVICES=0 python -u sft.py \
  --base_model="Qwen/Qwen2-1.5B" \
  --train_file="${train_file}" \
  --eval_file="${eval_file}" \
  --output_dir="output_dir/sft_qlora" \
  --category="${category}" \
  --sid_index_path="./data/Amazon/index/${category}.index.json" \
  --item_meta_path="./data/Amazon/index/${category}.item.json" \
  --sample=-1 \
  --num_epochs=1 \
  --cutoff_len=256 \
  --micro_batch_size=1 \
  --batch_size=4 \
  --learning_rate=1e-4 \
  --freeze_LLM=False \
  --use_qlora=True \
  --lora_r=8 \
  --lora_alpha=16 \
  --lora_dropout=0.05 \
  --lora_target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --bnb_4bit_quant_type="nf4" \
  --bnb_4bit_use_double_quant=True \
  --qlora_compute_dtype="bfloat16" \
  --group_by_length=False \
  --seed=42
