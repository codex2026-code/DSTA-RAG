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
