import os
import sys
import time
from typing import List
import numpy as np 
import fire
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import EarlyStoppingCallback, AutoConfig
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
import torch.nn as nn
import math
import warnings
from functools import partial
import numpy as np 
import fire
import transformers
from torch.optim.lr_scheduler import LambdaLR
import json
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from data import D3Dataset, SFTData, SidSFTDataset, SidItemFeatDataset, FusionSeqRecDataset, PreferenceSFTDataset, UserPreference2sidSFTDataset, TitleHistory2SidSFTDataset
import random
from datasets import Dataset as HFDataset
from torch.utils.data import ConcatDataset
from budget_utils import (
    BudgetStopCallback,
    JsonlMetricsCallback,
    apply_reproducible_subset,
    build_summary_from_log_history,
    ensure_dir,
    get_cuda_peak_memory_bytes,
    reset_cuda_peak_memory,
    write_summary_json,
)

# 从index文件中提取新token，并添加到tokenizer中
class TokenExtender:
    def __init__(self, data_path, dataset, index_file=".index.json"):
        self.data_path = data_path
        self.dataset = dataset
        self.index_file = index_file
        self.indices = None
        self.new_tokens = None
        
    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
    
    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens
            
        if self.indices is None:
            self._load_data()
        
        self.new_tokens = set()
        for index in self.indices.values(): #每一个SID对应一个index，index里包含了这个SID相关的token id列表
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))
        
        return self.new_tokens


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_target_modules(lora_target_modules):
    if isinstance(lora_target_modules, (list, tuple)):
        modules = [str(m).strip().strip("'\"") for m in lora_target_modules]
        return [m for m in modules if m]

    text = str(lora_target_modules).strip()
    if (text.startswith("(") and text.endswith(")")) or (text.startswith("[") and text.endswith("]")):
        text = text[1:-1]

    modules = []
    for module_name in text.split(","):
        module_name = module_name.strip().strip("'\"")
        if module_name:
            modules.append(module_name)
    return modules

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step, *, num_warmup_steps, num_training_steps, num_cycles
):
    if current_step < num_warmup_steps:
        return max(0.1, float(current_step) / float(max(1, num_warmup_steps)))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5, last_epoch: int = -1
):

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)



def train(
    # model/data params
    base_model: str = "",  # the only required argument
    train_file: str="",
    eval_file: str="",
    output_dir: str = "",
    sample: int = -1,
    seed: int = 42,
    
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    # llm hyperparams
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    freeze_LLM: bool = False,  # freeze LLM parameters, only train new token embeddings
    use_qlora: bool = False,  # enable QLoRA training (4-bit base + LoRA adapters)
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
    qlora_compute_dtype: str = "bfloat16",
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    category: str="",
    train_from_scratch: bool = False,
    sid_index_path: str = "",
    item_meta_path: str = "",
    # 固定预算与小样本闭环参数（用于 4GB 单卡快速 A/B 实验）
    max_train_samples: int = -1,
    train_subset_ratio: float = -1.0,
    max_eval_samples: int = -1,
    budget_hours: float = 0.0,
    budget_steps: int = -1,
):
    set_seed(seed)
    ensure_dir(output_dir)
    run_start_time = time.time()
    reset_cuda_peak_memory()
    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    summary_path = os.path.join(output_dir, "summary.json")

    os.environ['WANDB_PROJECT'] = wandb_project
    category_dict = {"Industrial_and_Scientific": "industrial and scientific items", "Office_Products": "office products", "Toys_and_Games": "toys and games", "Sports": "sports and outdoors", "Books": "books"}
    print(category)
    category = category_dict[category]
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1)) #如果没有设置WORLD_SIZE环境变量，则默认为1，表示单GPU训练
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if use_qlora and train_from_scratch:
        raise ValueError("QLoRA does not support --train_from_scratch=True. Please use a pretrained base model.")
    if use_qlora and freeze_LLM:
        raise ValueError("QLoRA cannot be combined with --freeze_LLM=True. Set freeze_LLM=False for QLoRA.")

    if use_qlora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        except ImportError as exc:
            raise ImportError(
                "QLoRA requires `peft`. Please install it (e.g. `pip install peft`)."
            ) from exc

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        qlora_compute_dtype = qlora_compute_dtype.lower()
        if qlora_compute_dtype not in dtype_map:
            raise ValueError(
                f"Unsupported qlora_compute_dtype={qlora_compute_dtype}. "
                "Choose from: bfloat16, float16, float32."
            )
        compute_dtype = dtype_map[qlora_compute_dtype]
        target_modules = parse_target_modules(lora_target_modules)
        if not target_modules:
            raise ValueError("lora_target_modules is empty. Please provide at least one module name.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=compute_dtype,
            quantization_config=quantization_config,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    elif not train_from_scratch:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
        )
    else: #这个从0训练一个大模型，还是算了
        config = AutoConfig.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_config(config)
        print("Training from scratch!")
        
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    if sid_index_path and os.path.exists(sid_index_path): #给了sid_index_path，就从index文件里提取新token，并添加到tokenizer和模型的embedding层中
        print(f"Loading index from {sid_index_path}")
        token_extender = TokenExtender(
            data_path=os.path.dirname(sid_index_path),
            dataset=os.path.basename(sid_index_path).split('.')[0]
        )
        new_tokens = token_extender.get_new_tokens()
        if new_tokens:
            print(f"Adding {len(new_tokens)} new tokens to tokenizer")
            # 加入一行记录原始词表大小，以便后续冻结参数时使用
            original_vocab_size = len(tokenizer)
            tokenizer.add_tokens(new_tokens)
            model.resize_token_embeddings(len(tokenizer))

    # Freeze LLM parameters if required
    if freeze_LLM and not use_qlora:
        print("Freezing LLM parameters, only training new token embeddings")
        for param in model.parameters():
            param.requires_grad = False

        if sid_index_path and os.path.exists(sid_index_path) and new_tokens:
            embedding_layer = model.get_input_embeddings()
            if embedding_layer.weight.shape[0] > original_vocab_size: #仅允许新增的embedding有梯度
                embedding_layer.weight.requires_grad = True

                def mask_grad(grad): #做一个梯度hook，强制将原始词表部分的梯度置零，只更新新增token的embedding
                    # grad shape: [vocab_size, hidden_dim]
                    grad[:original_vocab_size].zero_()
                    return grad
                
                embedding_layer.weight.register_hook(mask_grad)

                print(f"Unfrozen {len(new_tokens)} new token embeddings "
                    f"(indices {original_vocab_size} to {len(tokenizer)-1})")

        else:
            print("Warning: freeze_LLM=True but no new tokens added. All parameters are frozen!")

        # Print the number of trainable parameters (it will still report the size of the entire embedding matrix, but only the newly added rows will have non-zero gradients).
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params     = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters (with grad-mask): {trainable_params:,} / "
            f"{total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
    train_datasets = []
    # train_data1 = SFTData(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    train_data1 = SidSFTDataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    train_datasets.append(train_data1) #第一组任务，历史SID到下一个SID的预测，主要是让模型学会基本的序列推荐能力
    train_data2 = SidItemFeatDataset(item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    train_datasets.append(train_data2) #第二组任务，给模型看SID对应的item特征，让模型学会理解这些特征和SID之间的关系，这样模型就不只是记忆了SID的id，还能理解这个SID代表的物品是什么
    train_data3 = FusionSeqRecDataset(train_file=train_file, item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category)
    train_datasets.append(train_data3) #第三组任务，把历史SID和对应的item特征融合在一起，让模型学会同时利用序列信息和特征信息进行推荐，这样模型的推荐能力应该会更强一些
    # train_data4 = SFTData(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    # train_datasets.append(train_data4)
    # train_data5 = TitleHistory2SidSFTDataset(train_file=train_file, item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category)
    # train_datasets.append(train_data5)
    train_data = ConcatDataset(train_datasets) #把多组任务的数据集合在一起训练，这样模型就能同时学习这些不同的能力，提升泛化性能
    val_data = SidSFTDataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    # val_data = SFTData(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=20000, seed=seed, category=category)
    print("LOAD DATA FINISHED")    
    
    if resume_from_checkpoint:
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    
    sample_frac = 1
    hf_train_dataset = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})
    hf_train_dataset = hf_train_dataset.shuffle(seed=42).select(range(int(sample_frac * len(hf_train_dataset))))
    hf_val_dataset = HFDataset.from_dict({k: [v[k] for v in val_data] for k in val_data[0].keys()}).shuffle(seed=seed)
    hf_val_dataset = hf_val_dataset.shuffle(seed=42)

    # 小数据快速闭环：固定索引抽样并缓存到 output_dir，保证不同实验配置比较公平。
    hf_train_dataset, train_subset_meta = apply_reproducible_subset(
        hf_train_dataset,
        max_samples=max_train_samples,
        subset_ratio=train_subset_ratio,
        seed=seed,
        subset_index_path=os.path.join(output_dir, "train_subset_index.json"),
        subset_name="train",
    )
    hf_val_dataset, eval_subset_meta = apply_reproducible_subset(
        hf_val_dataset,
        max_samples=max_eval_samples,
        subset_ratio=-1.0,
        seed=seed,
        subset_index_path=os.path.join(output_dir, "eval_subset_index.json"),
        subset_name="eval",
    )

    print(hf_train_dataset)
    print(hf_val_dataset)
    eval_step = 0.05
    training_bf16 = True
    training_fp16 = False
    if use_qlora and qlora_compute_dtype == "float16":
        training_bf16 = False
        training_fp16 = True
    if use_qlora and qlora_compute_dtype == "float32":
        training_bf16 = False
        training_fp16 = False

    # 统一日志：每个 logging/eval 周期写入 metrics.jsonl，便于后续跨实验汇总。
    callbacks = [
        JsonlMetricsCallback(metrics_path=metrics_path, run_start_time=run_start_time),
        EarlyStoppingCallback(early_stopping_patience=3),
    ]
    if budget_hours and budget_hours > 0:
        callbacks.append(BudgetStopCallback(budget_hours=budget_hours))

    trainer = transformers.Trainer(
        # deepspeed=deepspeed,
        model=model,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        args=transformers.TrainingArguments(
            # deepspeed=deepspeed,
            run_name=wandb_run_name,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            max_steps=budget_steps if budget_steps > 0 else -1,
            learning_rate=learning_rate,
            bf16=training_bf16,
            fp16=training_fp16,
            logging_steps=1,
            optim="paged_adamw_8bit" if use_qlora else "adamw_torch",
            eval_strategy="steps",
            eval_steps=eval_step, 
            save_strategy="steps",
            save_steps=eval_step,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ), #把 padding 后的序列长度再“向上补齐”到 8 的倍数。返回tensors="pt"表示返回PyTorch的张量格式，padding=True表示对输入进行padding，使得同一批次内的序列长度一致，方便并行计算。
        callbacks=callbacks, #早停，如果连续3次评估指标没有提升，就停止训练，防止过拟合和节省计算资源
        # optimizers=(optimizer, lr_scheduler) 
    )
    model.config.use_cache = False #训练阶段不需要KV cache，关闭它可以节省显存和加速训练
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)

    if use_qlora:
        adapter_dir = os.path.join(output_dir, "final_adapter")
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        print(f"Saved QLoRA adapter to {adapter_dir}")

        merged_dir = os.path.join(output_dir, "final_checkpoint")
        try:
            merged_model = trainer.model.merge_and_unload()
            merged_model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            print(f"Saved merged full model to {merged_dir}")
        except Exception as e:
            print(f"Warning: failed to merge LoRA into base model: {e}")
            print("Adapter is available and can still be used for PEFT-based inference.")
    else:
        final_dir = os.path.join(output_dir, "final_checkpoint")
        trainer.model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)

    run_end_time = time.time()
    peak_memory_bytes = get_cuda_peak_memory_bytes()
    final_metrics, best_metrics = build_summary_from_log_history(trainer.state.log_history)
    runtime_sec = max(1e-8, run_end_time - run_start_time)
    summary = {
        "entry_script": "sft.py",
        "output_dir": output_dir,
        "seed": int(seed),
        "budget_hours": float(budget_hours),
        "budget_steps": int(budget_steps),
        "runtime_sec": float(runtime_sec),
        "global_steps": int(trainer.state.global_step),
        "throughput_steps_per_sec": float(trainer.state.global_step / runtime_sec),
        "peak_memory_bytes": int(peak_memory_bytes),
        "peak_memory_mb": float(peak_memory_bytes / (1024 ** 2)),
        "train_subset": train_subset_meta,
        "eval_subset": eval_subset_meta,
        "final_metrics": final_metrics,
        "best_metrics": best_metrics,
    }
    write_summary_json(summary_path, summary)
    print(f"Saved budget summary to {summary_path}")



if __name__ == "__main__":
    fire.Fire(train)
