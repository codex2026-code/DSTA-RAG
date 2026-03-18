# DSTA-RAG Codex Workspace

这个工作区实现了一个沿着 **Search-R1 → AutoRefine** 主线的最小侵入版本：

- **Stage 1**：用 `HF Transformers + TRL SFTTrainer + PEFT/LoRA` 学会 `assess → refine → rectify` 协议。
- **Stage 2**：在 **veRL / Search-R1 / AutoRefine** 风格的多轮检索环境中，用自定义 reward 函数做 RL 后训练。

本仓库不改 Search-R1/AutoRefine 的环境动作空间，仍然只把 `<search>` 和 `<answer>` 当作环境动作；
`<assess>`、`<refine>`、`<rectify>` 只作为中间协议块，由 parser、reward 和 evaluator 消费。

## 快速开始

### 1) 安装本仓库

```bash
pip install -e .
```

### 1.5) 从 Hugging Face 下载 HotpotQA(train) 并转换

```bash
bash cmd/prepare_hotpot_from_hf.sh \
  --split train \
  --raw-output examples/raw_hotpot_like.jsonl \
  --trace-output artifacts/stage1/traces.jsonl
```

### 2) Stage 1：构造 traces + SFT 数据 + 训练

```bash
bash cmd/build_stage1.sh \
  --input examples/raw_hotpot_like.jsonl \
  --trace-output artifacts/stage1/traces.jsonl \
  --sft-output artifacts/stage1/sft.jsonl

bash cmd/train_sft.sh \
  --config configs/stage1/qwen25_3b_lora.yaml \
  --train-file artifacts/stage1/sft.jsonl
```

> 若 7B 训练在 tokenizer 初始化时报错（提示需要 `sentencepiece` 或 `tiktoken`），可直接使用 `configs/stage1/qwen25_7b_lora.yaml` 中的 `tokenizer_use_fast: false`，或安装依赖：`pip install sentencepiece tiktoken`。


> 对全量 HotpotQA train split 做 SFT 时，建议优先采用 `max_steps` 限制训练步数（而不是拉长 epoch），
> 以降低过拟合风险并稳定协议学习。仓库默认 `configs/stage1/qwen25_3b_lora.yaml` 已采用该策略，
> 无需重新处理数据即可直接调参训练。


### 2.5) Stage 1 ckpt 指令遵循评测（协议合法性 + 动作完整性）

```bash
bash cmd/eval_stage1.sh \
  --model-path checkpoints/stage1/qwen25_3b_lora/checkpoint-3000 \
  --input examples/raw_hotpot_like.jsonl \
  --limit 100 \
  --output artifacts/stage1/eval_outputs.jsonl \
  --summary artifacts/stage1/eval_summary.json
```

评测会统计：`valid_rate`、`assess/refine/rectify/think` 出现率、终止动作完整率，以及 `<answer>` 的 EM/F1。

### 3) Stage 2：构造 parquet + 启动 veRL 训练

```bash
bash cmd/build_stage2.sh \
  --input examples/raw_hotpot_like.jsonl \
  --output-dir artifacts/stage2

bash cmd/train_rl.sh \
  --config configs/stage2/qwen25_3b_grpo.yaml \
  --train-file artifacts/stage2/train.parquet \
  --val-file artifacts/stage2/val.parquet \
  --autorefine-root /path/to/AutoRefine
```

> Stage 2 默认假设你已经有一个可运行的 AutoRefine / Search-R1 / veRL 工作目录；
> 本仓库通过 `custom_reward_function.path` 把 DSTA 奖励函数注入 veRL。

#### 3.1) 使用 W&B offline 监控 reward 组成项趋势

`dsta_rag.stage2.reward_fn.compute_score` 在保持返回总奖励 (`total`) 不变的同时，
支持按环境变量开关把 reward 各组成项写入 W&B（offline 模式可用）。

```bash
export WANDB_MODE=offline
export WANDB_DIR=artifacts/wandb_offline
export WANDB_PROJECT=DSTA-RAG
export WANDB_NAME=dsta-stage2-debug
export DSTA_WANDB_REWARD_LOG=1
```

然后按正常 Stage 2 命令启动训练。开启后会记录如下曲线：

- `reward/answer`
- `reward/coverage`
- `reward/ts`, `reward/it`, `reward/ta`
- `reward/faithfulness`
- `reward/stop`
- `reward/cost_penalty`
- `reward/validation_penalty`
- `reward/total`

默认仅 rank0 进程打点，避免分布式重复写日志。

## 目录

```text
cmd/
configs/
dsta_rag/
examples/
tests/
```

## 设计原则

1. **不改环境动作空间**：继续让 Search-R1/AutoRefine 处理 `<search>`/`<answer>`。
2. **先学协议，再做 RL**：Stage 1 只解决协议合法性和状态写入，Stage 2 再优化策略。
3. **先固定 retriever**：第一版不联训 retriever，避免把收益归因搞混。
4. **rule-based 过程奖励优先**：TS / IT / stop / coverage 先做弱监督代理，再逐步升级。

## 上游依赖

- Stage 1：`transformers`, `datasets`, `trl`, `peft`
- Stage 2：`verl`，以及 Search-R1/AutoRefine 风格的多轮检索 rollout 代码

如果你要在同一个工作区里组织 Codex 项目，建议把上游代码放到：

```text
third_party/AutoRefine
```

然后将 `--autorefine-root third_party/AutoRefine` 传给 `cmd/train_rl.sh`。
