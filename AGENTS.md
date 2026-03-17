# DSTA-RAG Codex Workspace

## Working agreements
- Stage 1 uses `cmd/build_stage1.sh` and `cmd/train_sft.sh`.
- Stage 2 uses `cmd/build_stage2.sh` and `cmd/train_rl.sh`.
- Keep the Search-R1/AutoRefine action space unchanged: only `<search>` and `<answer>` are environment actions.
- Treat `<assess>`, `<refine>`, and `<rectify>` as protocol blocks consumed by parsers, rewards, and offline evaluators.
- Prefer deterministic, rule-based proxies in the first implementation; do not add online LLM-judge calls unless a task explicitly asks for them.
- Do not modify upstream `search_r1/search/` or retriever code in the first implementation.
- Run `pytest tests/` after changing parser, validator, or reward logic.

## Repository layout
- `dsta_rag/stage1/`: trace construction, SFT dataset construction, SFT training.
- `dsta_rag/stage2/`: RL dataset construction, veRL launcher, custom reward function, replay evaluation.
- `configs/`: yaml configs for stage 1 and stage 2.
- `examples/`: minimal examples for schema inspection.
