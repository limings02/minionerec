# 基于 MiniOneRec 的生成式推荐系统复现与改造

MiniOneRec Reproduction and Engineering Improvements for Generative Recommendation

复现minionerec项目记录。

项目主线是把推荐任务改写成固定长度语义 token 的生成问题：先用 item 文本做表征，再做离散量化得到 semantic ID（SID），然后基于用户历史去生成下一个 item 的 SID，最后再用 constrained decoding 和 RL 做排序优化。

我做这个项目的原因也比较直接：想把“生成式推荐”从论文里的方法图，落到能训练、能评测、能定位问题的代码上。当前仓库重点覆盖离线复现、训练链路验证和几个我自己重点改过的模块，适合面试官快速看，也适合我自己面试时顺着讲 pipeline、难点和改造点。

## 30 秒看这个项目

- 这是什么：一个基于 MiniOneRec 的生成式推荐复现项目，不是原创提出 MiniOneRec。
- 我为什么做：想把 `item text -> SID -> SFT -> constrained decoding -> RL` 这条链路真正跑通，而不只停留在论文描述。
- 我具体做了什么：完成了从 item 表征、RQ-VAE 离散量化、SID 构建到生成式排序训练的端到端复现，并在 RQ-VAE 稳定性和 RL reward 设计上做了改造。
- 我改了什么：RQ-VAE 这边加了 K-means warm start、loss 协同和 Sinkhorn；RL 这边重写了 rank-aware reward 的一部分分配逻辑，处理 all-miss 和候选顺序问题。
- 结果怎么看：当前先放核心摘要，`collision rate -1.64%`，`HR@10 +2.3%`，`NDCG@10 +1.4%`。
- 先看哪些代码：[rq/models/vq.py](./rq/models/vq.py)、[rq/generate_indices.py](./rq/generate_indices.py)、[sft.py](./sft.py)、[rl.py](./rl.py)、[minionerec_trainer.py](./minionerec_trainer.py)、[evaluate.py](./evaluate.py)。

## 我做了什么 / My Contributions

### 1. 端到端复现生成式推荐链路

- 基于用户历史交互和 item 文本信息，把 `文本侧 item 表征 -> RQ-VAE 离散量化 -> SID 构建 -> SFT 微调 -> constrained decoding -> RL 微调` 这条链路串起来了。
- 把传统检索 / 排序式推荐任务改写成固定长度语义 token 的生成问题，最后还是用离线 `HR@K / NDCG@K` 来看结果。
- 数据侧补齐了从 `*.item.json / *.index.json / *.inter` 到训练 CSV、评测 JSON 的转换流程，便于重复实验和局部调试。

### 2. RQ-VAE / SID 构建阶段的稳定性优化

- 我在 [rq/models/vq.py](./rq/models/vq.py)、[rq/models/rq.py](./rq/models/rq.py)、[rq/models/rqvae.py](./rq/models/rqvae.py) 这条离散编码链路上，重点处理了 codebook 训练前期不稳定的问题。
- 加了 `K-means warm start` 初始化 codebook，避免随机初始化时前几轮量化中心很乱。
- reconstruction loss、codebook loss、commitment loss 这几个项是一起看的，不是只盯重建误差。
- 用 Sinkhorn 约束分配缓解 codebook collapse 和 early-stage 码本利用率不足，目的是减少碰撞、提升码本利用率。
- 这部分改造后，我记录到的一个直接结果是 `collision rate` 下降了 `1.64%`。

### 3. RL 奖励机制和训练逻辑改造

- [rl.py](./rl.py) 里我改了 rank-aware reward 的分配逻辑，主要是为了解决 reward 稀疏、all-miss 样本整组无反馈、候选顺序和奖励不一致这几类问题。

主要改动有四个：

- 调整 `rule_reward + ndcg_rule_reward` 的组合方式。
- 为 all-miss 样本补上有效反馈，避免一整组奖励全 0，导致 `reward_std=0`。
- 区分 invalid SID 和 valid-but-wrong SID，减少“错得不一样但全算一样”的问题。
- 修正非 beam search 场景下奖励和真实候选顺序不一致的问题。
- 这部分改造后，我目前记录到的离线提升是 `HR@10 +2.3%`、`NDCG@10 +1.4%`。

### 4. 训练 / 评测 / 实验脚本整理

- 给 SFT / RL 补了单卡可跑的 `QLoRA` 入口，保留多卡脚本的同时，也能在资源受限的环境下做闭环实验。
- 增加了固定预算实验相关的脚本和配置，包括 [budget_utils.py](./budget_utils.py)、[configs/experiments](./configs/experiments)、[scripts/run_budget_experiments.py](./scripts/run_budget_experiments.py)、[scripts/summarize_results.py](./scripts/summarize_results.py)。
- 评测阶段我保留了闭集 constrained decoding 这套逻辑，避免生成不存在的 item SID；单卡 4GB 的复现排障过程记录在 [REPRO_4GB.md](./REPRO_4GB.md) 和 [REPRO_4GB_STEP_BY_STEP.md](./REPRO_4GB_STEP_BY_STEP.md)。

## 方法链路 / Pipeline

### 1. Item text -> embedding

- 入口是 [rq/text2emb/amazon_text2emb.py](./rq/text2emb/amazon_text2emb.py)。
- 这里会读取 `*.item.json` 里的 `title + description`，生成 item embedding，并保存成 `*.emb-qwen-td.npy`。
- 当前仓库的 [data/Amazon/index](./data/Amazon/index) 下已经有现成的 embedding、index 和 item meta 文件，适合直接往后跑。

### 2. embedding -> RQ-VAE discretization

- 训练入口是 [rq/rqvae.py](./rq/rqvae.py)。
- 核心实现分别在 [rq/models/rqvae.py](./rq/models/rqvae.py)、[rq/models/rq.py](./rq/models/rq.py)、[rq/models/vq.py](./rq/models/vq.py)。
- `kmeans_init`、`beta`、`Sinkhorn` 都在这层实现里，和 codebook 稳定性直接相关。
- [rq/trainer.py](./rq/trainer.py) 训练时会同时看 loss 和 collision rate，并按 `best_loss_model.pth` / `best_collision_model.pth` 保存 ckpt。

### 3. discretization -> SID construction

- 入口是 [rq/generate_indices.py](./rq/generate_indices.py)。
- 这一步会把离散编码转成最终的 SID，并对 collision item 做迭代消解。
- 如果走 `RQ-KMeans+` 分支，对应入口是 [rq/rqkmeans_plus.py](./rq/rqkmeans_plus.py) 和 [rq/generate_indices_plus.py](./rq/generate_indices_plus.py)。
- 最终会生成 `*.index.json`，把 item id 映射到 `["<a_x>", "<b_y>", "<c_z>"]` 这样的 fixed-length semantic token。
- 需要注意的是，[rq/generate_indices.py](./rq/generate_indices.py) 目前还是实验脚本写法，文件头部的 `dataset / ckpt_path / output_dir` 需要先按自己的实验路径改一下，再执行。

### 4. SID -> SFT

- 入口是 [sft.py](./sft.py)。

当前实现里，SFT 不是只做一个 next-SID 任务，而是把三类数据拼在一起训练：

- [SidSFTDataset](./data.py)：历史 SID -> 下一个 SID。
- [SidItemFeatDataset](./data.py)：`sid <-> title / description` 对齐。
- [FusionSeqRecDataset](./data.py)：历史 SID + item 文本特征联合建模。
- [sft.py](./sft.py) 还负责从 `*.index.json` 中抽取新增 SID token，扩展 tokenizer 和 embedding，并支持 `freeze_LLM` / `use_qlora` 两种训练方式。

### 5. SFT -> RL fine-tuning

- 入口是 [rl.py](./rl.py)，底层 trainer 在 [minionerec_trainer.py](./minionerec_trainer.py)。

当前 RL 阶段混合了三类 prompt 来源：

- [SidDataset](./data.py)：历史 SID -> 目标 SID。
- [RLTitle2SidDataset](./data.py)：title / description -> SID。
- [RLSeqTitle2SidDataset](./data.py)：历史 title 序列 -> SID。
- [minionerec_trainer.py](./minionerec_trainer.py) 基于 TRL 的 GRPO trainer 做了推荐场景改造，把 constrained decoding、group reward 和推荐任务日志统计接进来了。

### 6. constrained decoding -> offline evaluation

- 评测入口是 [evaluate.py](./evaluate.py)，约束逻辑在 [LogitProcessor.py](./LogitProcessor.py)。
- 做法是先根据 `info_file` 里的合法 SID 构 prefix trie，然后在生成时只保留当前 prefix 下合法的 next token。
- 这一步很关键，因为如果不做闭集约束，生成结果很容易落到不存在的 item 上，`calc.py` 里的 `CC` 会非常难看。
- 最后用 [calc.py](./calc.py) 计算 `HR@K / NDCG@K / CC`。

## 仓库结构

```text
.
├── config/
│   └── zero2_opt.yaml
├── configs/
│   └── experiments/
├── data/
│   ├── Amazon/
│   │   ├── index/
│   │   ├── info/
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   ├── amazon18_data_process.py
│   ├── amazon23_data_process.py
│   └── process.py
├── rq/
│   ├── models/
│   ├── text2emb/
│   ├── rqvae.py
│   ├── rqkmeans_constrained.py
│   ├── rqkmeans_plus.py
│   ├── generate_indices.py
│   └── trainer.py
├── scripts/
│   ├── run_budget_experiments.py
│   └── summarize_results.py
├── output_dir/
├── results/
├── sft.py
├── rl.py
├── evaluate.py
├── minionerec_trainer.py
├── LogitProcessor.py
├── convert_dataset.py
├── calc.py
├── split.py
├── merge.py
├── run_sft.sh
├── run_sft_qlora.sh
├── run_rl_qlora.sh
├── REPRO_4GB.md
└── REPRO_4GB_STEP_BY_STEP.md
```

我自己平时最常看的几个位置：

- [config/zero2_opt.yaml](./config/zero2_opt.yaml)
  - RL 多卡启动时用的 accelerate / deepspeed 配置。
- [configs/experiments](./configs/experiments)
  - 固定预算实验的 YAML，里面有 `qlora_baseline / fast_gen / stable` 这些对照。
- [data](./data)
  - 原始数据清洗脚本在这里；[data/Amazon](./data/Amazon) 已经是当前可直接喂给训练和评测的格式。
- [rq](./rq)
  - item 文本编码、离散量化、SID 构建的核心代码。
- [sft.py](./sft.py)
  - SID token 扩表、多任务 SFT、freeze LLM / QLoRA 支持都在这里。
- [rl.py](./rl.py)
  - 奖励函数、数据拼接、GRPO 配置、debug fail-fast 都在这里。
- [evaluate.py](./evaluate.py) + [LogitProcessor.py](./LogitProcessor.py)
  - 离线生成 top-k 候选和闭集约束。
- [split.py](./split.py) + [merge.py](./merge.py) + [evaluate.sh](./evaluate.sh)
  - 多 GPU 评测时会用到。
- [debug_rl_smoke.py](./debug_rl_smoke.py)
  - 我排查 reward、LoRA 注入和 trainable params 时常用的一步 smoke 脚本。
- [output_dir](./output_dir) / [results](./results) / [wandb](./wandb)
  - 这些更多是实验产物和日志，不是主代码。

另外，仓库里还有 [sft_gpr.py](./sft_gpr.py)、[rl_gpr.py](./rl_gpr.py)、[convert_dataset_gpr.py](./convert_dataset_gpr.py) 这类支线实验文件。它们确实在仓库里，但不是我这份 README 重点讲的主线。

## 在 MiniOneRec 基础上的复现范围与改动

### 属于 MiniOneRec 原始方法的部分

- 用 item 文本构造语义表示。
- 用离散编码把 item 映射成 fixed-length SID。
- 用 LLM 根据用户历史生成下一个 SID 做推荐。
- 用 constrained decoding 保证生成结果落在合法 item 闭集。
- 用 RL 在 SFT 基础上继续优化排序效果。

### 我这份仓库重点做的部分

- 把这套方法补成能从数据到评测真正跑通的离线复现流程。
- 围绕 RQ-VAE 训练稳定性、码本利用率和 SID collision 做改造。
- 围绕 RL reward 稀疏、all-miss 无反馈、排序奖励错位做重构。
- 加入单卡 QLoRA、budget experiment、metrics 汇总和 debug 脚本，方便在有限算力下重复验证。

换句话说，这里强调的是“我基于 MiniOneRec 做了什么”，不是“我提出了 MiniOneRec”。

## 实验结果

这里先放最核心的结果摘要。完整对照表和日志我还在继续整理，所以先不把它写成论文式大表。

| 模块 / 改动 | 目标问题 | 当前记录到的变化 |
| --- | --- | --- |
| RQ-VAE + K-means warm start + Sinkhorn | codebook collapse、早期码本利用不足、SID collision | collision rate 下降 1.64% |
| RL rank-aware reward 重构 | reward 稀疏、all-miss 样本无反馈 | HR@10 提升 2.3% |
| RL 奖励分配修正 | 候选顺序与奖励不一致、排序信号失真 | NDCG@10 提升 1.4% |

如果你只想快速确认“这个仓库是不是实际跑过”，我建议直接看：

- [REPRO_4GB_STEP_BY_STEP.md](./REPRO_4GB_STEP_BY_STEP.md)
  - 里面是我在低显存环境下把 `evaluate.py -> calc.py` 整条链路真正跑通时留下的记录。
- [results/Industrial_ckpt_1gpu/final_result_Industrial_and_Scientific.json](./results/Industrial_ckpt_1gpu/final_result_Industrial_and_Scientific.json)
  - 这是仓库里现成保留的一份离线生成结果。

## 如何运行

这里只放我自己现在还在用的主入口，不写成长教程。

### 1. 环境

```bash
conda create -n minionerec python=3.11 -y
conda activate minionerec
pip install -r requirements.txt
```

### 2. 数据准备

如果你想从 Amazon18 原始数据开始：

```bash
python data/amazon18_data_process.py \
  --dataset Industrial_and_Scientific \
  --user_k 5 \
  --item_k 5 \
  --st_year 1996 \
  --st_month 10 \
  --ed_year 2018 \
  --ed_month 10 \
  --output_path ./Amazon18
```

如果只是想复现实验主线，当前仓库的 [data/Amazon](./data/Amazon) 已经包含 `train / valid / test / info / index` 这几个关键目录，可以直接往下走。现成整理好的主要是 `Industrial_and_Scientific` 和 `Office_Products` 两个品类。

### 3. item text -> embedding

```bash
python rq/text2emb/amazon_text2emb.py \
  --dataset Industrial_and_Scientific \
  --root /path/to/processed_dataset_dir \
  --plm_checkpoint your_emb_model_path
```

### 4. RQ-VAE / SID

```bash
python rq/rqvae.py \
  --data_path ./data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy \
  --ckpt_dir ./output/Industrial_and_Scientific \
  --lr 1e-3 \
  --epochs 10000 \
  --batch_size 20480
```

生成 SID：

```bash
python rq/generate_indices.py
```

这里再强调一下：`rq/generate_indices.py` 不是 argparse 风格，直接运行前要先改文件头部的路径常量。

如果你想看 `RQ-KMeans+` 或 constrained variant，对应入口在：

- [rq/rqkmeans_constrained.py](./rq/rqkmeans_constrained.py)
- [rq/rqkmeans_plus.py](./rq/rqkmeans_plus.py)
- [rq/generate_indices_plus.py](./rq/generate_indices_plus.py)

### 5. 数据转换

```bash
python convert_dataset.py \
  --dataset_name Industrial_and_Scientific \
  --data_dir /path/to/dataset_dir \
  --output_dir ./data/Amazon
```

### 6. SFT

单卡轻量入口：

```bash
bash run_sft.sh
```

QLoRA 入口：

```bash
bash run_sft_qlora.sh
```

原始多卡入口：

```bash
bash sft.sh
```

### 7. RL

单卡 QLoRA 入口：

```bash
MODEL_PATH=./Industrial_ckpt bash run_rl_qlora.sh
```

原始多卡入口：

```bash
bash rl.sh
```

### 8. Evaluate

```bash
python evaluate.py \
  --base_model ./Industrial_ckpt \
  --info_file ./data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt \
  --category Industrial_and_Scientific \
  --test_data_path ./data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --result_json_data ./results/Industrial_ckpt_1gpu/final_result_Industrial_and_Scientific.json \
  --batch_size 1 \
  --num_beams 3 \
  --max_new_tokens 64
```

计算指标：

```bash
python calc.py \
  --path ./results/Industrial_ckpt_1gpu/final_result_Industrial_and_Scientific.json \
  --item_path ./data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt
```

## 项目难点 / 复现踩坑

- `codebook collapse`
  - RQ-VAE 前期如果初始化和约束没处理好，码本很容易塌，后面 SID 冲突会一直难看。
- `early-stage codebook utilization`
  - 这也是我为什么更关注 warm start 和 Sinkhorn，而不是只盯 reconstruction loss。
- `invalid item generation`
  - 不做 constrained decoding，模型会生成不在 item 闭集里的 SID，`calc.py` 里的 `CC` 会直接暴露这个问题。
- `reward 稀疏`
  - RL 阶段最麻烦的不是 reward 公式怎么写，而是 all-miss 样本整组没信号时，训练其实很容易退化。
- `候选顺序和 reward 对不上`
  - 如果奖励分配和生成顺序没对齐，表面上是在优化排序，实际上可能是在给错候选发奖励。
- `训练稳定性和显存约束`
  - 仓库里保留单卡脚本和 budget configs，不是为了“好看”，主要是这部分真的是复现里最花时间的地方。

## Limitations / TODO

- 当前重点还是离线复现、训练链路验证和局部改造，不是线上服务化项目。
- README 里先放核心结果摘要，完整的对照实验表和日志整理还可以继续补。
- 仓库里已经有一些 GPR 相关支线脚本，但我当前这份项目说明还是以 MiniOneRec 主线复现为主。
- 当前数据集覆盖主要还是 Amazon 场景下的几个品类，后续可以再扩到更多公开数据集。
- [data.py](./data.py) 里还保留了一些实验期实现痕迹，比如多类 dataset 并存、字段解析方式比较粗糙，后面可以继续清理。

## Reference

- 这份仓库是基于 MiniOneRec 做的复现与工程化改造，不是原创提出 MiniOneRec。
- MiniOneRec 原始论文可以从旧版 README 里的链接找到，对应 arXiv 条目是 `2510.24431`。
