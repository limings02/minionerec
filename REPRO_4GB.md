你现在已经有所有材料了。我建议你在项目根目录新建一个 REPRO_4GB.md，内容按下面 6 段写（每段 3–6 行就行）：

目标与硬件约束（4GB 单卡）

数据/ckpt 准备（info/test/Industrial_ckpt）

必改点（禁用 GPU hardcode + max_len 参数化）

Sanity（30/300 条，CC=0）

Ablation（关约束 CC≈1032，invalid≈75%，指标掉）

全量结果（4533 条 HR/NDCG + CC=0，耗时约 27min）

4) 你现在必须能串起来的完整链路（闭环复述版）

info_file 提供合法 SID 列表（闭集 item）

每个 SID 拼成 ### Response:\n{SID}\n 后 tokenize 成 ID

对每个 ID 的每个 prefix，构造：
key = get_hash(prefix_tokens) → allowed_next_tokens.add(next_token)

推理时 ConstrainedLogitsProcessor 对每个 beam：

从 sent 里截取一段 token 作为 hash_key

查 prefix_allowed_tokens_fn 得到 allowed set

logits 里只保留 allowed token，其它 -inf

结果：生成永远在 trie 上走 → calc.py 映射永远成功 → CC=0


复现步骤（4GB 单卡）

环境：WSL2 + conda env + pip install -r requirements.txt

数据：data/Amazon/info/*.txt + data/Amazon/test/*.csv

模型：Industrial_ckpt/（含 config.json/tokenizer/model.safetensors）

必改点

禁用 evaluate.py 内部硬编码 CUDA_VISIBLE_DEVICES="0"（否则多进程/多卡无效）

将 EvalSidDataset(max_len=2560) 改为 --max_len 可配置（4GB 下建议 512/1024）

trie key 用 tuple(token_ids)（替代字符串 join，降低推理开销）

4GB 推荐推理参数

batch_size=1

num_beams=3（稳）或 5（略更好但更慢/更吃显存）

max_new_tokens=64

max_len=512

约束解码必要性（ablation）

开约束：CC=0（输出闭集可映射）

关约束：CC≈1032（calc 口径）/ invalid≈75%（全 beam 口径），HR/NDCG 明显下降

最后一块：你问的“一周能不能每步都知道为什么”

能。你现在已经把最难的部分（constrained decoding + trie）吃透了。剩下的“为什么”主要是两块：

SID 怎么来的（rq / residual quantization / codebook）：训练阶段的核心思想

EvalSidDataset 的 prompt 格式：为什么输入长、为什么 category 要转自然语言、为什么 history_item_sid 要 eval 成 list


你现在的数据侧已经全部讲通了（从 CSV 到指标）

给你一个最终“闭环地图”（你复盘用）：

CSVBaseDataset 读 CSV

EvalSidDataset.get_history()：解析 history_item_sid → 拼 prompt input；GT=item_sid

generate_prompt()：固定模板，关键锚点 ### Response:\n

pre(test=True)：只返回 prompt 的 input_ids / attention_mask

evaluate.py：batch 内 left-pad 到 maxLen，mask pad=0

generate()：beam search + ConstrainedLogitsProcessor（trie 约束）

sequences[:, maxLen:]：截出 completion，decode 成 SID 候选

写 json：每条样本加 predict

calc.py：用 output 对比 predict → HR/NDCG，CC 统计 OOV

最后一个你必须会的“质疑点”（你已经具备材料）

你能指出 repo 的一个“伪参数/未生效”的问题，并用数据证明它不影响当前复现：

K 传进来但没用

但 prompt token 最大 120，远小于 max_len=512，不会截断历史

所以当前评测不受影响；但换数据可能受影响，建议实现 K 截断或保留统计监控

这是非常典型的工程闭环叙事。