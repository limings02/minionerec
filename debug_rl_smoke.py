import argparse
import math
import os
import random

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import ConcatDataset
from transformers import BitsAndBytesConfig
from trl import GRPOConfig

from data import RLSeqTitle2SidDataset, RLTitle2SidDataset, SidDataset
from minionerec_trainer import ReReTrainer
from rl import (
    _debug_check_optimizer_param_groups,
    _debug_check_tokenizer_and_embeddings,
    _debug_check_trainable_params,
    parse_target_modules,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_reward_functions(
    prompt2history,
    history2target,
    valid_sid_set,
    num_generations: int,
    reward_type: str,
    reward_shaping: bool,
    invalid_reward: float,
    valid_non_target_reward: float,
    target_reward: float,
):
    ndcg_rewards = [-1.0 / math.log2(i + 2) for i in range(num_generations)]
    ndcg_rewards = [-elm / sum(ndcg_rewards) for elm in ndcg_rewards]

    def _safe_get_target(prompt: str) -> str:
        history = prompt2history.get(prompt, "")
        return history2target.get(history, "")

    def _normalize_sid(text: str) -> str:
        return str(text).split("Response:\n")[-1].strip().strip("\"")

    def rule_reward(prompts, completions):
        rewards = []
        for prompt, completion in zip(prompts, completions):
            target = _normalize_sid(_safe_get_target(prompt))
            completion_sid = _normalize_sid(completion)
            if completion_sid == target:
                rewards.append(float(target_reward))
            else:
                if reward_shaping:
                    if completion_sid not in valid_sid_set:
                        rewards.append(float(invalid_reward))
                    else:
                        rewards.append(float(valid_non_target_reward))
                else:
                    rewards.append(0.0)
        return rewards

    def ndcg_rule_reward(prompts, completions):
        rewards = []
        flag = False
        local_rewards = []
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            target = _normalize_sid(_safe_get_target(prompt))
            completion_sid = _normalize_sid(completion)
            if completion_sid == target:
                flag = True
                local_rewards.append(0.0)
            else:
                if reward_shaping and completion_sid not in valid_sid_set:
                    local_rewards.append(float(invalid_reward))
                else:
                    local_rewards.append(ndcg_rewards[i % num_generations])

            if (i + 1) % num_generations == 0:
                if flag:
                    rewards.extend(local_rewards)
                else:
                    if reward_shaping:
                        rewards.extend(local_rewards)
                    else:
                        rewards.extend([0.0] * num_generations)
                flag = False
                local_rewards = []
        return rewards

    if reward_type == "rule":
        return rule_reward
    if reward_type == "ranking":
        return [rule_reward, ndcg_rule_reward]
    if reward_type == "ranking_only":
        return ndcg_rule_reward
    raise ValueError(f"Unsupported reward_type for smoke script: {reward_type}")


def parse_args():
    parser = argparse.ArgumentParser(description="One-step RL smoke debug for MiniOneRec.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--info_file", type=str, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--sid_index_path", type=str, required=True)
    parser.add_argument("--item_meta_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output_dir/debug_smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_completion_length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--reward_type", type=str, default="ranking")
    parser.add_argument("--reward_shaping", action="store_true")
    parser.add_argument("--invalid_reward", type=float, default=-1.0)
    parser.add_argument("--valid_non_target_reward", type=float, default=0.0)
    parser.add_argument("--target_reward", type=float, default=1.0)

    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    parser.add_argument("--bnb_4bit_use_double_quant", dest="bnb_4bit_use_double_quant", action="store_true")
    parser.add_argument("--no_bnb_4bit_use_double_quant", dest="bnb_4bit_use_double_quant", action="store_false")
    parser.add_argument("--qlora_compute_dtype", type=str, default="bfloat16")

    parser.add_argument("--debug_fail_fast", dest="debug_fail_fast", action="store_true")
    parser.add_argument("--no_debug_fail_fast", dest="debug_fail_fast", action="store_false")
    parser.set_defaults(
        bnb_4bit_use_double_quant=True,
        debug_fail_fast=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["DEBUG_RL"] = "1"
    set_seed(args.seed)

    category_dict = {
        "Industrial_and_Scientific": "industrial and scientific items",
        "Office_Products": "office products",
        "Toys_and_Games": "toys and games",
        "Sports": "sports and outdoors",
        "Books": "books",
    }
    category_text = category_dict[args.category]

    train_datasets = []
    train_data1 = SidDataset(args.train_file, category=category_text, sample=-1)
    train_data2 = RLTitle2SidDataset(
        item_file=args.item_meta_path,
        index_file=args.sid_index_path,
        category=category_text,
        sample=-1,
    )
    train_data3 = RLSeqTitle2SidDataset(args.train_file, category=category_text, sample=10000)
    train_datasets.extend([train_data1, train_data2, train_data3])

    train_data = ConcatDataset(train_datasets)
    train_dataset = Dataset.from_dict({k: [elm[k] for elm in train_data] for k in train_data[0].keys()})
    train_dataset = train_dataset.shuffle(seed=args.seed)
    if len(train_dataset) == 0:
        raise RuntimeError("Empty train_dataset in smoke script.")

    prompt2history = {}
    history2target = {}
    for dataset in train_datasets:
        if hasattr(dataset, "prompt2history"):
            prompt2history.update(dataset.prompt2history)
        if hasattr(dataset, "history2target"):
            history2target.update(dataset.history2target)

    valid_sid_set = set()
    with open(args.info_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split("\t")
            if parts:
                valid_sid_set.add(parts[0].strip())

    reward_fun = build_reward_functions(
        prompt2history=prompt2history,
        history2target=history2target,
        valid_sid_set=valid_sid_set,
        num_generations=args.num_generations,
        reward_type=args.reward_type,
        reward_shaping=args.reward_shaping,
        invalid_reward=args.invalid_reward,
        valid_non_target_reward=args.valid_non_target_reward,
        target_reward=args.target_reward,
    )

    model_init_kwargs = {"torch_dtype": torch.bfloat16}
    peft_config = None
    if args.use_qlora:
        from peft import LoraConfig, TaskType

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        compute_dtype = dtype_map[args.qlora_compute_dtype.lower()]
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model_init_kwargs = {
            "torch_dtype": compute_dtype,
            "quantization_config": quantization_config,
        }
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=parse_target_modules(args.lora_target_modules),
        )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        eval_strategy="no",
        save_strategy="no",
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=1,
        logging_steps=1,
        learning_rate=1e-6,
        beta=args.beta,
        num_train_epochs=1,
        max_steps=1,
        model_init_kwargs=model_init_kwargs,
        report_to=[],
        run_name="debug_smoke",
    )

    trainer = ReReTrainer(
        model=args.model_path,
        base_model=args.model_path,
        reward_funcs=reward_fun,
        train_dataset=train_dataset,
        eval_dataset=None,
        args=training_args,
        add_gt=False,
        dynamic_sampling=False,
        beam_search=False,
        test_during_training=False,
        info_file=args.info_file,
        prompt2history=prompt2history,
        history2target=history2target,
        peft_config=peft_config,
        debug=True,
        debug_fail_fast=args.debug_fail_fast,
        debug_short_completion_patience=10,
        debug_reward_std_patience=10,
        debug_print_every_steps=1,
    )

    _debug_check_tokenizer_and_embeddings(
        tokenizer=trainer.processing_class,
        model=trainer.model,
        model_path=args.model_path,
        sid_index_path=args.sid_index_path,
        debug_fail_fast=args.debug_fail_fast,
    )
    _debug_check_trainable_params(
        model=trainer.model,
        use_qlora=args.use_qlora,
        debug_fail_fast=args.debug_fail_fast,
    )
    _debug_check_optimizer_param_groups(
        trainer=trainer,
        debug_fail_fast=args.debug_fail_fast,
    )

    first_sample = train_dataset[0]
    mini_batch = [dict(first_sample) for _ in range(args.num_generations)]
    print(
        "[DEBUG][SMOKE] "
        f"mini_batch_size={len(mini_batch)}, num_generations={args.num_generations}, "
        f"prompt_preview={repr(first_sample.get('prompt', ''))[:120]}"
    )

    trainer.model.eval()
    with torch.no_grad():
        prepared = trainer._prepare_inputs(mini_batch)
        loss = trainer.compute_loss(trainer.model, prepared)

    completion_mask = prepared["completion_mask"]
    mean_completion_length = completion_mask.sum(1).float().mean().item()
    print(
        "[DEBUG][SMOKE] "
        f"single_step_loss={float(loss.item()):.8f}, "
        f"mean_completion_length={mean_completion_length:.4f}"
    )

    metric_snapshot = {k: (sum(v) / len(v) if len(v) > 0 else None) for k, v in trainer._metrics.items()}
    interested = [
        "reward",
        "reward_std",
        "invalid_sid_rate",
        "duplicate_rate",
        "completion_length",
        "token_diversity",
        "kl",
    ]
    for key in interested:
        if key in metric_snapshot:
            print(f"[DEBUG][SMOKE] {key}={metric_snapshot[key]}")

    print("[DEBUG][SMOKE] done")


if __name__ == "__main__":
    main()
