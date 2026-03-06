from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import random
import numpy as np
import torch
from data import D3Dataset, SidDataset, RLTitle2SidDataset, RLSeqTitle2SidDataset, RLSid2TitleDataset, RLSidhis2TitleDataset
from torch.utils.data import ConcatDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import time
from pathlib import Path
from minionerec_trainer import ReReTrainer
from sasrec import SASRec
from fire import Fire
import pickle
import math
import json
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

os.environ['WANDB_MODE'] = 'disabled'

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


def _env_truthy(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_debug_mode(debug: bool) -> bool:
    return bool(debug) or _env_truthy("DEBUG_RL", default="0")


def _extract_sid_probe_tokens(
    sid_index_path: str,
    fallback_tokens: list[str] = None,
) -> list[str]:
    if fallback_tokens is None:
        fallback_tokens = ["<a_0>", "<b_0>", "<c_0>"]
    if not sid_index_path or not os.path.exists(sid_index_path):
        return fallback_tokens

    try:
        with open(sid_index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
    except Exception:
        return fallback_tokens

    tokens = []
    for value in index_data.values():
        if not isinstance(value, list):
            continue
        for token in value:
            if isinstance(token, str) and token not in tokens:
                tokens.append(token)
            if len(tokens) >= 3:
                return tokens
    return fallback_tokens


def _debug_check_model_path_artifacts(model_path: str) -> None:
    local_path = Path(model_path)
    if not local_path.exists():
        print(f"[DEBUG][TOKEN] model_path={model_path} (not a local path, maybe hub id)")
        return

    tokenizer_candidates = [
        local_path / "tokenizer.json",
        local_path / "tokenizer.model",
        local_path / "vocab.json",
        local_path / "special_tokens_map.json",
    ]
    has_tokenizer_files = any(path.exists() for path in tokenizer_candidates)
    has_adapter = (local_path / "adapter_config.json").exists()
    has_config = (local_path / "config.json").exists()
    print(
        "[DEBUG][TOKEN] "
        f"model_path={model_path}, has_config={has_config}, "
        f"has_tokenizer_files={has_tokenizer_files}, has_adapter_config={has_adapter}"
    )


def _debug_check_tokenizer_and_embeddings(
    *,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    model_path: str,
    sid_index_path: str,
    debug_fail_fast: bool = True,
) -> None:
    _debug_check_model_path_artifacts(model_path)

    print(
        "[DEBUG][TOKEN] "
        f"eos_token_id={tokenizer.eos_token_id}, "
        f"pad_token_id={tokenizer.pad_token_id}, "
        f"unk_token_id={tokenizer.unk_token_id}, "
        f"vocab_size={len(tokenizer)}"
    )

    probe_tokens = _extract_sid_probe_tokens(sid_index_path)
    probe_ids = []
    bad_tokens = []
    for token in probe_tokens[:3]:
        token_id = tokenizer.convert_tokens_to_ids(token)
        probe_ids.append(int(token_id) if token_id is not None else -1)
        print(f"[DEBUG][TOKEN] probe_token={token}, token_id={token_id}")
        if token_id is None or int(token_id) < 0:
            bad_tokens.append((token, token_id, "invalid_id"))
        elif tokenizer.unk_token_id is not None and int(token_id) == int(tokenizer.unk_token_id):
            bad_tokens.append((token, token_id, "unk_id"))

    if len(set(probe_ids)) < len(probe_ids):
        duplicated = {}
        for token, token_id in zip(probe_tokens[:3], probe_ids):
            duplicated.setdefault(token_id, []).append(token)
        duplicated = {k: v for k, v in duplicated.items() if len(v) > 1}
        message = (
            "[DEBUG][TOKEN][FAILFAST] SID probe tokens map to duplicated ids: "
            f"{duplicated}. tokenizer 可能未包含 SID tokens，或 model_path 指向了 base model。"
        )
        if debug_fail_fast:
            raise RuntimeError(message)
        print(message)

    if bad_tokens:
        message = (
            "[DEBUG][TOKEN][FAILFAST] SID probe tokens unresolved/UNK: "
            f"{bad_tokens}. tokenizer 未包含 SID tokens，或 model_path 错误。"
        )
        if debug_fail_fast:
            raise RuntimeError(message)
        print(message)

    embedding_rows = int(model.get_input_embeddings().weight.shape[0])
    tokenizer_len = int(len(tokenizer))
    print(
        f"[DEBUG][TOKEN] embedding_rows={embedding_rows}, tokenizer_len={tokenizer_len}"
    )
    if embedding_rows != tokenizer_len:
        message = (
            "[DEBUG][TOKEN][FAILFAST] model embedding size != tokenizer size. "
            f"embedding_rows={embedding_rows}, tokenizer_len={tokenizer_len}. "
            "请确认 checkpoint 包含 SID tokenizer，并在需要时执行 resize_token_embeddings。"
        )
        if debug_fail_fast:
            raise RuntimeError(message)
        print(message)


def _debug_check_trainable_params(
    *,
    model: AutoModelForCausalLM,
    use_qlora: bool,
    debug_fail_fast: bool = True,
) -> None:
    trainable_params = 0
    trainable_names = []
    has_lora_param = False
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        trainable_params += int(param.numel())
        if len(trainable_names) < 20:
            trainable_names.append(name)
        if "lora" in name.lower():
            has_lora_param = True

    print(f"[DEBUG][LORA] trainable_params={trainable_params}")
    print(f"[DEBUG][LORA] first_trainable_names={trainable_names}")

    if trainable_params <= 0:
        message = "[DEBUG][LORA][FAILFAST] trainable_params=0，QLoRA/LoRA 没有挂上或参数被冻结。"
        if debug_fail_fast:
            raise RuntimeError(message)
        print(message)

    if use_qlora and not has_lora_param:
        message = (
            "[DEBUG][LORA][FAILFAST] use_qlora=True 但未检测到任何 lora_* 可训练参数；"
            "请检查 peft_config 是否真正注入。"
        )
        if debug_fail_fast:
            raise RuntimeError(message)
        print(message)


def _debug_check_optimizer_param_groups(
    trainer: ReReTrainer,
    *,
    debug_fail_fast: bool = True,
) -> None:
    if trainer.optimizer is None:
        trainer.create_optimizer()
    optimizer = trainer.optimizer
    if optimizer is None:
        message = "[DEBUG][LORA][FAILFAST] optimizer 为空，无法执行训练更新。"
        if debug_fail_fast:
            raise RuntimeError(message)
        print(message)
        return

    group_count = len(optimizer.param_groups)
    param_tensor_count = sum(len(group.get("params", [])) for group in optimizer.param_groups)
    print(
        "[DEBUG][LORA] "
        f"optimizer_group_count={group_count}, optimizer_param_tensor_count={param_tensor_count}"
    )
    if group_count <= 0 or param_tensor_count <= 0:
        message = (
            "[DEBUG][LORA][FAILFAST] optimizer.param_groups 为空；"
            "这会导致 grad_norm=0 / loss 无更新。"
        )
        if debug_fail_fast:
            raise RuntimeError(message)
        print(message)

def train(
    # model/data params
    model_path: str = "",
    seed: int = 42,
    train_file: str = "",
    eval_file: str = "",
    info_file: str = "",
    category: str = "",
    
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    
    # training hyperparams
    output_dir: str = "",
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    temperature: float = 1.0,
    add_gt: bool = False,
    eval_step: float = 0.199,
    num_generations: int = 16,
    max_completion_length: int = 128,
    num_train_epochs: int = 1,
    learning_rate: float = 1e-6,
    beta: float = 0.04,
    beam_search: bool = False,
    test_during_training: bool = True,
    dynamic_sampling: bool = False,
    mask_all_zero: bool = False,
    sync_ref_model: bool = False,
    test_beam: int = 20,
    reward_type: str = "rule",
    reward_shaping: bool = False,
    invalid_reward: float = -1.0,
    valid_non_target_reward: float = 0.0,
    target_reward: float = 1.0,
    sample_train: bool = False,
    ada_path: str = "",
    cf_path: str = "",
    sid_index_path: str = "",
    item_meta_path: str = "",
    dapo: bool = False,
    gspo: bool = False,
    use_qlora: bool = False,  # enable QLoRA training (4-bit base + LoRA adapters)
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
    qlora_compute_dtype: str = "bfloat16",
    # 固定预算与小样本闭环参数
    max_train_samples: int = -1,
    train_subset_ratio: float = -1.0,
    max_eval_samples: int = -1,
    budget_hours: float = 0.0,
    budget_steps: int = -1,
    # Debug 诊断开关（可通过 --debug 或环境变量 DEBUG_RL=1 启用）
    debug: bool = False,
    debug_fail_fast: bool = True,
    debug_short_completion_patience: int = 10,
    debug_reward_std_patience: int = 10,
    debug_print_every_steps: int = 1,
):
    torch.backends.cuda.enable_flash_sdp(False)  
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    set_seed(seed)
    debug_mode = _resolve_debug_mode(debug)
    if debug_mode:
        print("[DEBUG][TOKEN] DEBUG_RL mode enabled.")
    ensure_dir(output_dir)
    run_start_time = time.time()
    reset_cuda_peak_memory()
    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    summary_path = os.path.join(output_dir, "summary.json")

    model_init_kwargs = None
    peft_config = None
    llm_model_load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    training_bf16 = True
    training_fp16 = False
    optim_name = "paged_adamw_32bit"

    if use_qlora:
        try:
            from peft import LoraConfig, TaskType
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
        llm_model_load_kwargs = {
            "torch_dtype": compute_dtype,
            "device_map": "auto",
            "quantization_config": quantization_config,
        }
        model_init_kwargs = {
            "torch_dtype": compute_dtype,
            "quantization_config": quantization_config,
        }
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )
        optim_name = "paged_adamw_8bit"
        if qlora_compute_dtype == "float16":
            training_bf16 = False
            training_fp16 = True
        elif qlora_compute_dtype == "float32":
            training_bf16 = False
            training_fp16 = False
    
    category_dict = {"Industrial_and_Scientific": "industrial and scientific items", "Office_Products": "office products", "Toys_and_Games": "toys and games", "Sports": "sports and outdoors", "Books": "books"}
    print(category)
    
    
    with open(info_file, 'r') as f:
        info = f.readlines()
        # Extract semantic_id (first column) from the format: semantic_id \t item_title \t item_id
        item_name = [_.split('\t')[0].strip() for _ in info]
        item2id = {name: i for i, name in enumerate(item_name)}

    sample = -1
    train_datasets = []
    # train_data = D3Dataset(train_file, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data)
    train_data1 = SidDataset(train_file, category=category_dict[category], sample=sample)
    train_datasets.append(train_data1)
    train_data2 = RLTitle2SidDataset(item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    train_datasets.append(train_data2)
    train_data3 = RLSeqTitle2SidDataset(train_file, category=category_dict[category], sample=10000)
    train_datasets.append(train_data3)
    # train_data4 = RLSid2TitleDataset(item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data4)
    # train_data5 = RLSidhis2TitleDataset(train_file, item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data5)
    # train_data6 = RLTitle2Sid_1LayerDataset(item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data6)
    # train_data7 = RLTitle2Sid_2LayerDataset(item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data7)
    train_data = ConcatDataset(train_datasets)
    # eval_data = D3Dataset(eval_file, category=category_dict[category], sample=sample)
    eval_data = SidDataset(eval_file, category=category_dict[category], sample=sample)

    train_dataset = Dataset.from_dict({k : [elm[k] for elm in train_data] for k in train_data[0].keys()})
    train_dataset = train_dataset.shuffle(seed=seed) 
    if sample_train and "sft" in model_path:
        train_dataset = train_dataset.select(range(int(0.2 * len(train_dataset)), len(train_dataset)))
    eval_dataset = Dataset.from_dict({k : [elm[k] for elm in eval_data] for k in eval_data[0].keys()})
    eval_dataset = eval_dataset.shuffle(seed=seed)

    # 固定索引子集：保证不同实验共享同一批样本，便于预算对比。
    train_dataset, train_subset_meta = apply_reproducible_subset(
        train_dataset,
        max_samples=max_train_samples,
        subset_ratio=train_subset_ratio,
        seed=seed,
        subset_index_path=os.path.join(output_dir, "train_subset_index.json"),
        subset_name="train",
    )
    eval_dataset, eval_subset_meta = apply_reproducible_subset(
        eval_dataset,
        max_samples=max_eval_samples,
        subset_ratio=-1.0,
        seed=seed,
        subset_index_path=os.path.join(output_dir, "eval_subset_index.json"),
        subset_name="eval",
    )
    

    # prompt2history = {**train_data.prompt2history, **eval_data.prompt2history}
    # history2target = {**train_data.history2target, **eval_data.history2target}

    prompt2history = {}
    history2target = {}
    
    # Collect prompt2history and history2target from all train datasets
    for dataset in train_datasets:
        if hasattr(dataset, 'prompt2history'):
            prompt2history.update(dataset.prompt2history)
        if hasattr(dataset, 'history2target'):
            history2target.update(dataset.history2target)
    
    # Add eval_data mappings
    if hasattr(eval_data, 'prompt2history'):
        prompt2history.update(eval_data.prompt2history)
    if hasattr(eval_data, 'history2target'):
        history2target.update(eval_data.history2target)

    print("train_dataset: ", train_dataset)
    print("eval_dataset: ", eval_dataset)

    llm_model = AutoModelForCausalLM.from_pretrained(model_path, **llm_model_load_kwargs)
    device = llm_model.device
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if debug_mode:
        _debug_check_tokenizer_and_embeddings(
            tokenizer=tokenizer,
            model=llm_model,
            model_path=model_path,
            sid_index_path=sid_index_path,
            debug_fail_fast=debug_fail_fast,
        )
    
    len_seq = 10
    item_num = len(item_name)
    print(f"item_num: {item_num}")

    if reward_type == "sasrec":
        model = SASRec(32, item_num, len_seq, 0.3, device)
        model.to(device)
        model.load_state_dict(torch.load(cf_path))
        model.eval()
    if reward_type == "semantic":
        with open(ada_path, "rb") as f:
            item_ada_embd = pickle.load(f)
        item_ada_embd = torch.tensor(item_ada_embd).to(llm_model.device)

    print("Load item_ada_embd successfully.")
    if reward_shaping:
        print(
            "[DEBUG][REWARD] reward_shaping enabled: "
            f"invalid_reward={invalid_reward}, "
            f"valid_non_target_reward={valid_non_target_reward}, "
            f"target_reward={target_reward}"
        )

    ndcg_rewards = [-1.0/math.log2(i+2) for i in range(num_generations)]
    ndcg_rewards = [-elm/sum(ndcg_rewards) for elm in ndcg_rewards]

    def _normalize_sid(text: str) -> str:
        return str(text).split("Response:\n")[-1].strip().strip("\"")


    def ndcg_rule_reward(prompts, completions):
        history = [prompt2history[prompt] for prompt in prompts]
        targets = [history2target[elm] for elm in history]
        repeat = num_generations
        rewards = []
        flag = False
        lis = []

        for i, completion in enumerate(completions):
            normalized_completion = _normalize_sid(completion)
            normalized_target = _normalize_sid(targets[i])

            if normalized_completion == normalized_target:
                flag = True
                # ranking reward 保持原语义：命中项本身给 0，依赖 rule_reward 给主奖励。
                lis.append(0.0)
            else:
                if reward_shaping and normalized_completion not in item2id:
                    lis.append(float(invalid_reward))
                else:
                    lis.append(ndcg_rewards[i%num_generations])
            
            if (i+1)%num_generations == 0:
                if flag:
                    rewards.extend(lis)
                else:
                    # 原逻辑：无命中整组给 0（会导致 reward_std=0）
                    # shaping 开启后：保留 rank-based 负奖励，避免优势函数退化。
                    if reward_shaping:
                        rewards.extend(lis)
                    else:
                        rewards.extend([0.0] * repeat)
                flag = False
                lis = []
        
        return rewards

    def rule_reward(prompts, completions):
        history = [prompt2history[prompt] for prompt in prompts]
        targets = [history2target[elm] for elm in history]
        rewards = []

        for i, completion in enumerate(completions):
            normalized_completion = _normalize_sid(completion)
            normalized_target = _normalize_sid(targets[i])
            if normalized_completion == normalized_target:
                rewards.append(float(target_reward))
            else:
                if reward_shaping:
                    if normalized_completion not in item2id:
                        rewards.append(float(invalid_reward))
                    else:
                        rewards.append(float(valid_non_target_reward))
                else:
                    rewards.append(0.0)
        return rewards

    def semantic_reward(prompts, completions):
        history = [prompt2history[prompt] for prompt in prompts]
        targets = [history2target[elm] for elm in history]
        target_ids = [item2id[elm.strip("\"\n")] for elm in targets]
        completions = [elm.strip("\"\n") for elm in completions]
        for i, completion in enumerate(completions):
            if completion not in item2id:
                print("==============================")
                print(prompts[i])
                print(f"Invalid item: {completion}")
                print("==============================")
        completion_ids = [item2id[elm] for elm in completions]
        rewards =  torch.cosine_similarity(item_ada_embd[target_ids], item_ada_embd[completion_ids], dim=-1)
        print(rewards)
        return rewards

    def cf_reward(prompts, completions):
        history = [prompt2history[prompt] for prompt in prompts]
        history_list = [elm.split("::") for elm in history]
        pred_ids = []
        for i, elm in enumerate(completions):
            elm = elm.strip("\n\"")
            if elm not in item_name:
                # print("========Invalid Item========")
                # print(f"Invalid item: {elm}")
                # print(f"Prompt: {prompts[i]}")
                # print("============================")
                pred_ids.append(random.randint(0, item_num-1))
            else:
                pred_ids.append(item2id[elm])
        
        len_lis = []
        history_ids = []
        for his in history_list:
            his = [item2id[elm] for elm in his]
            len_lis.append(len(his))
            if len(his) < len_seq: 
                his = his + [item_num] * (len_seq - len(his))
            history_ids.append(his)
        
        seq = torch.LongTensor(history_ids).to(device)
        pred = torch.LongTensor(pred_ids).to(device)    
        
        with torch.no_grad():
            predictions = model.forward_eval(seq, torch.tensor(np.array(len_lis)).to(device))
            scores = torch.gather(predictions, 1,  pred.view(-1, 1)).view(-1)
        return scores
    


    if reward_type == "rule":
        reward_fun = rule_reward
    elif reward_type == "ranking":
        reward_fun = [rule_reward, ndcg_rule_reward]
    elif reward_type == "ranking_only":
        reward_fun = ndcg_rule_reward
    elif reward_type == "semantic":
        reward_fun = semantic_reward
    elif reward_type == "sasrec":
        reward_fun = cf_reward
    
    os.environ['WANDB_PROJECT'] = wandb_project
    os.environ["WANDB_MODE"] = "offline"

    callbacks = [
        JsonlMetricsCallback(metrics_path=metrics_path, run_start_time=run_start_time),
    ]
    if budget_hours and budget_hours > 0:
        callbacks.append(BudgetStopCallback(budget_hours=budget_hours))

    training_args = GRPOConfig(output_dir=output_dir,
                                save_steps=0.1,
                                save_total_limit=20,
                                eval_strategy="steps",
                                max_completion_length=max_completion_length,
                                num_generations=num_generations,
                                temperature=temperature,
                                sync_ref_model=sync_ref_model,
                                per_device_eval_batch_size=eval_batch_size,
                                per_device_train_batch_size=train_batch_size,
                                gradient_accumulation_steps=gradient_accumulation_steps,  
                                eval_steps=eval_step, 
                                logging_steps=1, 
                                learning_rate=learning_rate,
                                beta=beta,
                                warmup_ratio=0.03,
                                max_grad_norm= 0.3,
                                num_train_epochs=num_train_epochs,
                                max_steps=budget_steps if budget_steps > 0 else -1,
                                bf16=training_bf16,
                                fp16=training_fp16,
                                optim=optim_name,
                                model_init_kwargs=model_init_kwargs,
                                lr_scheduler_type="cosine", 
                                save_strategy="steps",
                                report_to="wandb",
                                run_name=wandb_run_name,
                            )
    trainer = ReReTrainer(
        model=model_path,
        base_model=model_path,
        dapo=dapo,
        gspo=gspo,
        add_gt=add_gt,
        dynamic_sampling=dynamic_sampling,
        beam_search=beam_search,
        test_during_training=test_during_training,
        test_beam=test_beam,
        info_file=info_file,
        prompt2history=prompt2history,
        history2target=history2target,
        reward_funcs=reward_fun,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        callbacks=callbacks,
        peft_config=peft_config,
        debug=debug_mode,
        debug_fail_fast=debug_fail_fast if debug_mode else False,
        debug_short_completion_patience=debug_short_completion_patience,
        debug_reward_std_patience=debug_reward_std_patience,
        debug_print_every_steps=debug_print_every_steps,
    )

    if debug_mode:
        _debug_check_trainable_params(
            model=trainer.model,
            use_qlora=use_qlora,
            debug_fail_fast=debug_fail_fast,
        )
        _debug_check_optimizer_param_groups(
            trainer=trainer,
            debug_fail_fast=debug_fail_fast,
        )

    trainer.train()

    trainer.save_model(output_dir)

    final_ckpt_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(final_ckpt_dir)
    tokenizer.save_pretrained(final_ckpt_dir)

    run_end_time = time.time()
    peak_memory_bytes = get_cuda_peak_memory_bytes()
    final_metrics, best_metrics = build_summary_from_log_history(trainer.state.log_history)
    runtime_sec = max(1e-8, run_end_time - run_start_time)
    summary = {
        "entry_script": "rl.py",
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
    Fire(train)
