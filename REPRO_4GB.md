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