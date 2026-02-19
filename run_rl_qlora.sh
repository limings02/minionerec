#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:64}"
source "$(dirname "$0")/wsl_proxy.sh" || true

category="${CATEGORY:-Industrial_and_Scientific}"
model_path="${MODEL_PATH:-path_to_model}"
output_dir="${OUTPUT_DIR:-output_dir/rl_qlora}"
wandb_run_name="${WANDB_RUN_NAME:-rl_qlora_demo}"
num_processes="${NUM_PROCESSES:-1}"
main_process_port="${MAIN_PROCESS_PORT:-29503}"
hf_endpoint="${HF_ENDPOINT:-https://hf-mirror.com}"

# 4GB-friendly defaults. All can be overridden by env vars.
train_batch_size="${TRAIN_BATCH_SIZE:-1}"
eval_batch_size="${EVAL_BATCH_SIZE:-2}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-8}"
num_train_epochs="${NUM_TRAIN_EPOCHS:-1}"
eval_step="${EVAL_STEP:-1.0}"
reward_type="${REWARD_TYPE:-ranking}"
num_generations="${NUM_GENERATIONS:-2}"
max_completion_length="${MAX_COMPLETION_LENGTH:-64}"
sync_ref_model="${SYNC_REF_MODEL:-False}"
beam_search="${BEAM_SEARCH:-False}"
temperature="${TEMPERATURE:-1.0}"
learning_rate="${LEARNING_RATE:-1e-5}"
beta="${BETA:-1e-3}"

lora_r="${LORA_R:-8}"
lora_alpha="${LORA_ALPHA:-16}"
lora_dropout="${LORA_DROPOUT:-0.05}"
lora_target_modules="${LORA_TARGET_MODULES:-q_proj,v_proj}"
bnb_4bit_quant_type="${BNB_4BIT_QUANT_TYPE:-nf4}"
bnb_4bit_use_double_quant="${BNB_4BIT_USE_DOUBLE_QUANT:-True}"
qlora_compute_dtype="${QLORA_COMPUTE_DTYPE:-float16}"

if [[ "${model_path}" == "path_to_model" ]]; then
  echo "Please set MODEL_PATH to a real base model path or HF repo id." >&2
  echo "Example: MODEL_PATH=./Industrial_ckpt bash run_rl_qlora.sh" >&2
  exit 1
fi

train_file=$(ls ./data/Amazon/train/${category}*.csv 2>/dev/null | head -1 || true)
eval_file=$(ls ./data/Amazon/valid/${category}*11.csv 2>/dev/null | head -1 || true)
info_file=$(ls ./data/Amazon/info/${category}*.txt 2>/dev/null | head -1 || true)

if [[ -z "${train_file}" || -z "${eval_file}" || -z "${info_file}" ]]; then
  echo "Failed to find train/eval/info files for category=${category}" >&2
  exit 1
fi

global_train_batch=$((train_batch_size * gradient_accumulation_steps * num_processes))
global_eval_batch=$((eval_batch_size * num_processes))
if (( global_train_batch % num_generations != 0 )); then
  echo "Invalid config: global train batch (${train_batch_size} * ${gradient_accumulation_steps} * ${num_processes} = ${global_train_batch}) must be divisible by NUM_GENERATIONS=${num_generations}." >&2
  echo "Try adjusting TRAIN_BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS / NUM_GENERATIONS." >&2
  exit 1
fi
if (( global_eval_batch % num_generations != 0 )); then
  echo "Invalid config: global eval batch (${eval_batch_size} * ${num_processes} = ${global_eval_batch}) must be divisible by NUM_GENERATIONS=${num_generations}." >&2
  echo "Try setting EVAL_BATCH_SIZE to a value that satisfies: EVAL_BATCH_SIZE * NUM_PROCESSES % NUM_GENERATIONS == 0." >&2
  exit 1
fi

accelerate_args=(
  launch
  --mixed_precision bf16
  --num_machines 1
  --dynamo_backend no
  --num_processes "${num_processes}"
  --main_process_port "${main_process_port}"
)
if [[ "${num_processes}" -gt 1 ]]; then
  accelerate_args+=(--multi_gpu)
fi

HF_ENDPOINT="${hf_endpoint}" accelerate "${accelerate_args[@]}" rl.py \
  --model_path "${model_path}" \
  --train_batch_size "${train_batch_size}" \
  --eval_batch_size "${eval_batch_size}" \
  --num_train_epochs "${num_train_epochs}" \
  --gradient_accumulation_steps "${gradient_accumulation_steps}" \
  --train_file "${train_file}" \
  --eval_file "${eval_file}" \
  --info_file "${info_file}" \
  --category "${category}" \
  --sample_train False \
  --eval_step "${eval_step}" \
  --reward_type "${reward_type}" \
  --num_generations "${num_generations}" \
  --max_completion_length "${max_completion_length}" \
  --mask_all_zero False \
  --dynamic_sampling False \
  --sync_ref_model "${sync_ref_model}" \
  --beam_search "${beam_search}" \
  --test_during_training False \
  --temperature "${temperature}" \
  --learning_rate "${learning_rate}" \
  --add_gt False \
  --beta "${beta}" \
  --dapo False \
  --output_dir "${output_dir}" \
  --wandb_run_name "${wandb_run_name}" \
  --sid_index_path "./data/Amazon/index/${category}.index.json" \
  --item_meta_path "./data/Amazon/index/${category}.item.json" \
  --use_qlora True \
  --lora_r "${lora_r}" \
  --lora_alpha "${lora_alpha}" \
  --lora_dropout "${lora_dropout}" \
  --lora_target_modules "${lora_target_modules}" \
  --bnb_4bit_quant_type "${bnb_4bit_quant_type}" \
  --bnb_4bit_use_double_quant "${bnb_4bit_use_double_quant}" \
  --qlora_compute_dtype "${qlora_compute_dtype}"
