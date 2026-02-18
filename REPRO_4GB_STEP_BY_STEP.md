REPRO_4GB_STEP_BY_STEP.md（可直接复制）
0. 目标与验收标准

**目标：**在本机（Windows + WSL2，RTX 3050 4GB）上跑通 MiniOneRec 的 Industrial_and_Scientific 品类评测，得到最终 final_result_*.json 并用 calc.py 输出指标。

验收：

生成结果文件：
results/Industrial_ckpt_1gpu/final_result_Industrial_and_Scientific.json

计算指标成功输出 HR/NDCG

CC=0（无非法输出：生成结果都能映射到物品库）

1. 环境准备（推荐 WSL2）
1.1 检查 WSL2 与 GPU

在 PowerShell：

wsl -l -v


进入 WSL 后：

lsb_release -a
nvidia-smi
df -h /


期望：

Ubuntu 22.04.x

nvidia-smi 能看到 RTX3050（4GB）

根分区空间足够（你这里显示 955G 可用，OK）

2. conda 环境（Miniconda）
2.1 安装后注意事项（非常常见坑）

如果你看到：

conda: command not found

说明你还没让 shell 加载 conda 初始化脚本。解决方式之一：

重新打开一个 WSL 终端

或执行：

source ~/.bashrc

2.2 ToS 报错（你已遇到）

当出现 CondaToSNonInteractiveError 时：

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

2.3 创建与激活环境
conda create -n minionerec python=3.11 -y
conda activate minionerec
python --version
which python


期望看到类似：

Python 3.11.x

python 路径在 .../envs/minionerec/bin/python

3. 拉代码与安装依赖

进入仓库根目录（示例）：

cd ~/projects/MiniOneRec
pip install -r requirements.txt

4. 数据与模型检查（必须确认三样都在）
4.1 数据文件
ls -lah ./data/Amazon/info/Industrial_and_Scientific*.txt
ls -lah ./data/Amazon/test/Industrial_and_Scientific*11.csv


你当前已有：

./data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt

./data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv

4.2 模型 ckpt（Industrial_ckpt）

你用的是软链接（OK）：

ls -lah Industrial_ckpt | head -50
ls -lah Industrial_ckpt/ | head -50


期望至少包含：

config.json

tokenizer.json / vocab.json / merges.txt

model.safetensors

5. 4GB 单卡必须改的两处（否则会踩坑）
5.1 禁用 evaluate.py 内部 GPU 硬编码

在 evaluate.py 里你已经做过：

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


原因：

否则你外部设置的 CUDA_VISIBLE_DEVICES=0 会被脚本覆盖

多进程/多卡脚本（如 evaluate.sh）会失效或行为异常

5.2 使用 --max_len 控制输入长度（4GB 稳定跑）

你已将 EvalSidDataset(..., max_len=...) 参数化，CLI 支持：

--max_len 512

你已经验证：

max_len=512 与更大值指标一致（因为 prompt token 最大约 120）

6. 先跑小样本 sanity（30 条）
6.1 运行 evaluate（30条）
mkdir -p temp/4gb

CUDA_VISIBLE_DEVICES=0 python -u ./evaluate.py \
  --base_model "./Industrial_ckpt" \
  --info_file "./data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt" \
  --category "Industrial_and_Scientific" \
  --test_data_path "temp/4gb/test_30.csv" \
  --result_json_data "temp/4gb/pred_30.json" \
  --batch_size 1 \
  --num_beams 3 \
  --max_new_tokens 64 \
  --max_len 512 \
  --length_penalty 0.0

6.2 计算指标（30条）
python ./calc.py \
  --path "temp/4gb/pred_30.json" \
  --item_path "./data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt"


你的 sanity 输出（记录在这里，作为验收）：

beams=3

HR@1=0.13333333, HR@3=0.13333333

NDCG@1=0.13333333, NDCG@3=0.13333333

CC=0

7. 全量评测（4533 条，单卡）
7.1 运行 evaluate（全量）

注意：命令续行符 \ 后面不能有空格，否则 fire 会报 “Could not consume arg”。

mkdir -p results/Industrial_ckpt_1gpu

CUDA_VISIBLE_DEVICES=0 python -u ./evaluate.py \
  --base_model "./Industrial_ckpt" \
  --info_file "./data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt" \
  --category "Industrial_and_Scientific" \
  --test_data_path "./data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
  --result_json_data "results/Industrial_ckpt_1gpu/final_result_Industrial_and_Scientific.json" \
  --batch_size 1 \
  --num_beams 3 \
  --max_new_tokens 64 \
  --max_len 512 \
  --length_penalty 0.0

7.2 计算指标（全量）
python ./calc.py \
  --path "results/Industrial_ckpt_1gpu/final_result_Industrial_and_Scientific.json" \
  --item_path "./data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt"


你的全量输出（写入这里，作为最终验收）：

beams=3

HR@1 = 0.08625634

HR@3 = 0.10941981

NDCG@1 = 0.08625634

NDCG@3 = 0.09985993

CC = 0

8. 常见报错与排查（强烈建议保留）
8.1 Could not consume arg: --temperature

现象：跑完很久后 fire 报错不识别 --temperature

原因通常有两个：

evaluate.py main() 没有 temperature 参数（你当前代码确实没有）

命令行续行符 \ 后带了空格，导致参数被拼坏（你踩过）

解决：

删除 --temperature/--guidance_scale 等不在签名中的参数

确保每行结尾是 \ 且 后面没有任何字符（含空格）

8.2 SettingWithCopyWarning

来自：

row['history_item_sid'] = eval(row['history_item_sid'])

影响：

不影响复现正确性（只是 pandas 警告）

工程修复建议：

用 ast.literal_eval

不要写回 row，用局部变量 hist = literal_eval(...)

8.3 CC 很大 / 指标明显掉

几乎必然是 没启用 constrained decoding。

检查点：

evaluate.py 是否构造并传入：

ConstrainedLogitsProcessor(...)

logits_processor=LogitsProcessorList([clp])

9. 4GB 推荐参数（你的机器验证可用）

batch_size=1

num_beams=3（速度/效果平衡）

max_new_tokens=64

max_len=512

length_penalty=0.0