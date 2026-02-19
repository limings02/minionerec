# QLoRA 改造说明与改动清单

## 1. 目标
在不破坏现有训练流程的前提下，为 `sft.py` 增加可选 QLoRA 微调能力。

兼容性原则：
- 默认行为不变：`--use_qlora=False` 时仍走原训练路径。
- 仅当显式开启 `--use_qlora=True` 才进入 QLoRA 分支。
- 保留原保存目录约定：仍生成 `final_checkpoint`（QLoRA 下为 merge 后模型）。

---

## 2. 文件改动总览
- `sft.py`：新增 QLoRA 参数、4bit+LoRA 加载逻辑、训练精度与优化器切换、QLoRA 保存逻辑。
- `requirements.txt`：新增 `peft` 依赖。
- `run_sft_qlora.sh`：新增一键 QLoRA 启动脚本（不改原 `run_sft.sh`）。

---

## 3. 逐项改动（行号 + 功能）

### A. `sft.py`

1. `sft.py:23`  
改动：`from transformers import ... BitsAndBytesConfig`  
功能：支持构建 4bit 量化配置（QLoRA 必需）。

2. `sft.py:108-115`  
改动：新增参数  
- `use_qlora`
- `lora_r`
- `lora_alpha`
- `lora_dropout`
- `lora_target_modules`
- `bnb_4bit_quant_type`
- `bnb_4bit_use_double_quant`
- `qlora_compute_dtype`  
功能：将 QLoRA 完整配置暴露为命令行参数，同时默认关闭，保持兼容。

3. `sft.py:142-145`  
改动：新增配置冲突校验  
- `use_qlora` 不能与 `train_from_scratch=True` 同时使用  
- `use_qlora` 不能与 `freeze_LLM=True` 同时使用  
功能：提前阻止无效组合，避免训练中途异常。

4. `sft.py:147-192`  
改动：新增 QLoRA 主分支  
- 动态导入 `peft`（`LoraConfig/TaskType/get_peft_model/prepare_model_for_kbit_training`）  
- `qlora_compute_dtype` 解析与合法性校验  
- `lora_target_modules` 字符串解析与非空校验  
- 构建 `BitsAndBytesConfig(load_in_4bit=True, ...)`  
- 4bit 方式加载模型  
- `prepare_model_for_kbit_training(...)`  
- 注入 LoRA（`get_peft_model(...)`）并打印可训练参数  
功能：完成 QLoRA 的完整模型初始化流程。

5. `sft.py:193-201`  
改动：将原有加载逻辑改为 `elif not train_from_scratch` / `else`  
功能：保证只有非 QLoRA 模式才走原全参/从零训练分支。

6. `sft.py:223`  
改动：`if freeze_LLM` -> `if freeze_LLM and not use_qlora`  
功能：避免冻结逻辑错误影响 QLoRA 训练。

7. `sft.py:287-294`  
改动：新增训练精度开关变量  
- `training_bf16`
- `training_fp16`
- 按 `qlora_compute_dtype` 自动设置  
功能：QLoRA 可根据配置切换 bf16/fp16/fp32 训练开关。

8. `sft.py:309-310`  
改动：`TrainingArguments` 中 `bf16` 固定值改为动态变量，并新增 `fp16`  
功能：与 QLoRA 精度配置保持一致。

9. `sft.py:312`  
改动：`optim="adamw_torch"` -> `optim="paged_adamw_8bit" if use_qlora else "adamw_torch"`  
功能：QLoRA 下启用更省显存的分页 8bit AdamW。

10. `sft.py:335-353`  
改动：重写训练后保存逻辑  
- QLoRA：先存 `final_adapter`，再尝试 `merge_and_unload()` 存 `final_checkpoint`  
- 非 QLoRA：保持原 `final_checkpoint` 保存逻辑  
功能：兼顾 PEFT 适配器保存与原评估脚本可直接加载的完整模型保存。

---

### B. `requirements.txt`

11. `requirements.txt:127`  
改动：新增 `peft==0.17.1`  
功能：显式声明 QLoRA 所需依赖，减少环境差异导致的导入失败。

---

### C. `run_sft_qlora.sh`

12. `run_sft_qlora.sh:1-34`（新增文件）  
改动：新增独立 QLoRA 启动脚本，包含：
- `--use_qlora=True`
- LoRA 超参
- 4bit 量化参数
- `--freeze_LLM=False`  
功能：不改动 `run_sft.sh` 的前提下，提供一键 QLoRA 训练入口。

---

## 4. 行为变化总结

### 4.1 不开启 QLoRA（默认）
- 行为与原先一致：模型加载、优化器、保存路径逻辑不变。

### 4.2 开启 QLoRA
- 以 4bit 加载底座模型并注入 LoRA；
- 优化器切换为 `paged_adamw_8bit`；
- 产物包含：
  - `output_dir/.../final_adapter`（LoRA adapter）
  - `output_dir/.../final_checkpoint`（merge 后完整模型，推荐用于现有评估脚本）

---

## 5. 使用方式

### 方式 1：直接用新增脚本
```bash
bash run_sft_qlora.sh
```

### 方式 2：在原命令中手动追加
```bash
--use_qlora=True \
--freeze_LLM=False \
--lora_r=8 \
--lora_alpha=16 \
--lora_dropout=0.05 \
--lora_target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
--bnb_4bit_quant_type="nf4" \
--bnb_4bit_use_double_quant=True \
--qlora_compute_dtype="bfloat16"
```

---

## 6. 已执行的基础校验
- `python -m py_compile sft.py`：通过
- `bash -n run_sft_qlora.sh`：通过

说明：尚未在该文档中包含完整训练跑通日志（需实际启动训练任务验证显存与收敛表现）。

---

## 7. RL 侧 QLoRA 改造说明与改动清单

### 7.1 目标
在不破坏现有 RL 训练流程的前提下，为 `rl.py` 增加可选 QLoRA 微调能力。

兼容性原则：
- 默认行为不变：`--use_qlora=False` 时仍走原 RL 训练路径。
- 仅当显式开启 `--use_qlora=True` 才进入 QLoRA 分支。
- 不改现有 `rl.sh`，通过追加参数即可启用 QLoRA。

---

### 7.2 文件改动总览
- `rl.py`：新增 QLoRA 参数、4bit 量化与 LoRA 配置、训练精度/优化器动态切换、将 `peft_config` 传入 `ReReTrainer`。

---

### 7.3 逐项改动（行号 + 功能）

1. `rl.py:8`  
改动：`from transformers import ... BitsAndBytesConfig`  
功能：支持构建 4bit 量化配置（QLoRA 必需）。

2. `rl.py:69-76`  
改动：新增参数  
- `use_qlora`
- `lora_r`
- `lora_alpha`
- `lora_dropout`
- `lora_target_modules`
- `bnb_4bit_quant_type`
- `bnb_4bit_use_double_quant`
- `qlora_compute_dtype`  
功能：将 RL 侧 QLoRA 开关与核心超参暴露为命令行参数，默认关闭以保持兼容。

3. `rl.py:82-142`  
改动：新增 QLoRA 初始化主分支  
- 动态导入 `peft`（`LoraConfig/TaskType`）  
- `qlora_compute_dtype` 解析与合法性校验  
- `lora_target_modules` 字符串解析与非空校验  
- 构建 `BitsAndBytesConfig(load_in_4bit=True, ...)`  
- 组装 `model_init_kwargs` 与 `peft_config`  
- 按 dtype 自动切换 `bf16/fp16`  
- 优化器在 QLoRA 下切换为 `paged_adamw_8bit`  
功能：完成 RL 训练时 QLoRA 所需的量化+LoRA 配置准备。

4. `rl.py:206`  
改动：`AutoModelForCausalLM.from_pretrained(...)` 改为使用 `llm_model_load_kwargs`。  
功能：在 QLoRA 模式下按 4bit 配置加载模型；非 QLoRA 时保持原 `bf16 + device_map=auto`。

5. `rl.py:351-354`  
改动：`GRPOConfig` 中  
- `bf16` 改为动态变量  
- 新增 `fp16`  
- `optim` 改为动态变量  
- 新增 `model_init_kwargs`  
功能：确保 `ReReTrainer` 内部按 QLoRA 配置再次加载模型时参数一致。

6. `rl.py:377`  
改动：`ReReTrainer(...)` 新增 `peft_config=peft_config`。  
功能：在训练器内自动将模型包装为 LoRA 训练模型；不开启 QLoRA 时该参数为 `None`，行为不变。

---

### 7.4 行为变化总结

#### 7.4.1 不开启 QLoRA（默认）
- 行为与原先一致：模型加载、优化器、训练精度配置、保存逻辑均保持原路径。

#### 7.4.2 开启 QLoRA
- 以 4bit 方式加载底座模型并注入 LoRA；
- `GRPOConfig` 与 `ReReTrainer` 接收一致的量化/PEFT配置；
- 优化器切换为 `paged_adamw_8bit`，显存占用更低。

---

### 7.5 运行例子（RL + QLoRA）

```bash
category="Industrial_and_Scientific"
train_file=$(ls -f ./data/Amazon/train/${category}*.csv)
eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
info_file=$(ls -f ./data/Amazon/info/${category}*.txt)

HF_ENDPOINT=https://hf-mirror.com accelerate launch \
  --config_file ./config/zero2_opt.yaml \
  --num_processes 8 --main_process_port 29503 \
  rl.py \
  --model_path path_to_model \
  --train_batch_size 64 \
  --eval_batch_size 128 \
  --num_train_epochs 2 \
  --gradient_accumulation_steps 2 \
  --train_file ${train_file} \
  --eval_file ${eval_file} \
  --info_file ${info_file} \
  --category ${category} \
  --reward_type ranking \
  --num_generations 16 \
  --sync_ref_model True \
  --learning_rate 1e-5 \
  --beta 1e-3 \
  --output_dir output_dir/rl_qlora \
  --wandb_run_name rl_qlora_demo \
  --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
  --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json \
  --use_qlora True \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --bnb_4bit_quant_type nf4 \
  --bnb_4bit_use_double_quant True \
  --qlora_compute_dtype bfloat16
```

---

### 7.6 已执行的基础校验（RL）
- `python -m py_compile rl.py`：通过
