```mermaid
flowchart LR
  %% =========================
  %% 1) SID Construction
  %% =========================
  subgraph S1["① SID Construction（把 item 变成离散 token：SID）"]
    A1["Raw Item Text\n(title + description)"]
    A2["Frozen Text Encoder\n(e.g., sentence encoder)"]
    A3["Dense Embedding\nitem_emb ∈ R^d"]
    A4["RQ Quantizer (3-level)\nRQ-VAE / RQ-Kmeans(+)\ncodebook_a,b,c"]
    A5["SID Tokens\n<a_i><b_j><c_k>"]
    A6["Artifacts\nsid_index.json / item_meta.json / info_file.txt\n(合法 SID 列表/前缀)"]
    A1 --> A2 --> A3 --> A4 --> A5 --> A6
  end

  %% =========================
  %% 2) Convert Dataset
  %% =========================
  subgraph S2["② Convert Dataset（交互数据变成 SFT/RL 可用格式）"]
    B1["Interaction CSV\n(user_id, history, target_item, ...)"]
    B2["Join SID Mapping\n(item_id → item_sid)"]
    B3["SFT Samples\ninput_ids + labels\n(prompt label=-100 mask)\n(history_sid → next_sid / sid↔title / fusion)"]
    B4["RL Samples\nprompt (+ groundtruth)\n(不提供 labels)\n用于生成 multiple completions"]
    B1 --> B2
    B2 --> B3
    B2 --> B4
  end

  %% =========================
  %% 3) SFT
  %% =========================
  subgraph S3["③ SFT（监督 next-token：先学会“会推荐”）"]
    C1["Load Base LLM\n(base/instruct)"]
    C2["Extend Tokenizer\n加入 <a_x>/<b_x>/<c_x> SID 词表"]
    C3["Optional: Freeze Base\n只训练新增 SID embedding\n(grad mask / hook)"]
    C4["Multi-task SFT\nseq-rec + sid↔title + fusion\nTrainer: next-token loss\n只算答案区 loss"]
    C5["SFT Checkpoint\npolicy_0"]
    C1 --> C2 --> C3 --> C4 --> C5
  end

  %% =========================
  %% 4) RL (GRPO)
  %% =========================
  subgraph S4["④ RL (GRPO)（在“会推荐”基础上对齐指标）"]
    D1["Init Policy = SFT ckpt\nπθ"]
    D2["Reference Model (fixed)\nπ_ref\n用于 KL 约束"]
    D3["Constrained Decoding\nTrie/Prefix Tree from info_file\n保证生成 SID 合法&去重\n(constrained beam search)"]
    D4["Generate Group\n每个 prompt → num_generations 候选"]
    D5["Reward\nrule / ranking / semantic / sasrec...\n组内归一化 advantage"]
    D6["Update\nGRPO objective + KL penalty\n得到 πθ'"]
    D7["RL Checkpoint\npolicy_rl"]
    D1 --> D3 --> D4 --> D5 --> D6 --> D7
    D2 --> D6
  end

  %% =========================
  %% 5) Eval
  %% =========================
  subgraph S5["⑤ Offline Eval（约束生成→算 HR/NDCG，并检查合法性）"]
    E1["Constrained Decode\n生成 Top-K SID / item"]
    E2["Metrics\nHR@K, NDCG@K"]
    E3["Validity Check\nCC!=0 ⇒ invalid 多\n(常见原因：依赖版本/约束解码失败)"]
    E1 --> E2
    E1 --> E3
  end

  %% =========================
  %% Cross-stage connections
  %% =========================
  S1 --> S2 --> S3 --> S4 --> S5

```

## Labels 对齐可视化（3 条真实样本）

数据来源：`temp/labels_visual_3samples.txt`  
记号约定：`I=instruction`、`P=prompt`、`T=target`。  
训练规则：`label=-100` 不参与 loss；`label=input_id` 参与 loss。

### 1) 总览

| Dataset | 样本来源 | full_len | I_len | P_len | T_len | split(I+P) | 可训练区间 |
|---|---|---:|---:|---:|---:|---:|---|
| `SidSFTDataset` | `train.csv` row0 | 85 | 44 | 36 | 5 | 80 | `pos 80..84` |
| `SidItemFeatDataset` | `item_id=0` (`sid2title`) | 82 | 38 | 19 | 25 | 57 | `pos 57..81` |
| `FusionSeqRecDataset` | `train.csv` row0 | 103 | 46 | 38 | 19 | 84 | `pos 84..102` |

### 2) 边界切换（`-100` -> 可训练）

| Dataset | pos | seg | input_id | input_tok | label | train |
|---|---:|:---:|---:|---|---:|---:|
| `SidSFTDataset` | 78 | P | 5949 | `\u0120Response` | -100 | 0 |
| `SidSFTDataset` | 79 | P | 510 | `:\u010a` | -100 | 0 |
| `SidSFTDataset` | 80 | T | 151666 | `<a_104>` | 151666 | 1 |
| `SidSFTDataset` | 81 | T | 151812 | `<b_118>` | 151812 | 1 |
| `SidSFTDataset` | 82 | T | 152126 | `<c_176>` | 152126 | 1 |
| `SidItemFeatDataset` | 55 | P | 5949 | `\u0120Response` | -100 | 0 |
| `SidItemFeatDataset` | 56 | P | 510 | `:\u010a` | -100 | 0 |
| `SidItemFeatDataset` | 57 | T | 75932 | `SUP` | 75932 | 1 |
| `SidItemFeatDataset` | 58 | T | 8281 | `CO` | 8281 | 1 |
| `SidItemFeatDataset` | 59 | T | 328 | `\u0120S` | 328 | 1 |
| `FusionSeqRecDataset` | 82 | P | 5949 | `\u0120Response` | -100 | 0 |
| `FusionSeqRecDataset` | 83 | P | 510 | `:\u010a` | -100 | 0 |
| `FusionSeqRecDataset` | 84 | T | 6464 | `Gr` | 6464 | 1 |
| `FusionSeqRecDataset` | 85 | T | 72725 | `izzly` | 72725 | 1 |
| `FusionSeqRecDataset` | 86 | T | 479 | `\u0120G` | 479 | 1 |

### 3) Target 区（全部 `train=1`）示例

| Dataset | target 文本（节选） | target token ids（节选） |
|---|---|---|
| `SidSFTDataset` | `<a_104><b_118><c_176>\n` | `151666, 151812, 152126, 198, 151643` |
| `SidItemFeatDataset` | `SUPCO SPP6 Relay/Capacitor ... Torque\n` | `75932, 8281, 328, 4406, ..., 591, 198, 151643` |
| `FusionSeqRecDataset` | `Grizzly G9849 Magnetic Base/Dial Indicator Combo - President's Special\n` | `6464, 72725, 479, 24, ..., 9785, 198, 151643` |

### 4) 一句话结论

三类样本完全一致：`instruction+prompt` 全部 `label=-100`；从 `split` 起进入 `target` 区，逐 token 监督学习。
