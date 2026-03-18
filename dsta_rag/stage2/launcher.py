from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sanitize_tokenizer_config(model_path: str) -> None:
    """
    Fix known tokenizer_config compatibility issues for local checkpoints.

    Some checkpoints are saved with `extra_special_tokens` as a JSON list, while
    recent `transformers` expects a dict-like mapping and calls `.keys()` on it.
    """
    path = Path(model_path)
    if not path.exists() or not path.is_dir():
        return

    tokenizer_cfg = path / "tokenizer_config.json"
    if not tokenizer_cfg.exists():
        return

    with open(tokenizer_cfg, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    extra_special_tokens = cfg.get("extra_special_tokens")
    if not isinstance(extra_special_tokens, list):
        return

    fixed = {}
    for idx, token in enumerate(extra_special_tokens):
        if isinstance(token, str) and token:
            fixed[f"extra_token_{idx}"] = token

    if fixed and "additional_special_tokens" not in cfg:
        cfg["additional_special_tokens"] = list(fixed.values())
    cfg["extra_special_tokens"] = fixed

    backup = tokenizer_cfg.with_suffix(".json.bak")
    if not backup.exists():
        tokenizer_cfg.replace(backup)
    with open(tokenizer_cfg, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(
        "[DSTA-RAG] Patched tokenizer_config.json: "
        "`extra_special_tokens` list -> dict for transformers compatibility."
    )


def validate_model_config(model_path: str) -> None:
    """
    Validate local model checkpoint config before launching AutoRefine/veRL.

    veRL stage-2 expects a decoder-only causal LM checkpoint path in
    `actor_rollout_ref.model.path`. If users accidentally point to a checkpoint
    directory with `model_type=rag`, transformers will instantiate `RagConfig`
    and fail later with a cryptic error about missing `question_encoder` and
    `generator`.
    """
    path = Path(model_path)
    if not path.exists() or not path.is_dir():
        return

    config_path = path / "config.json"
    if not config_path.exists():
        return

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_type = str(cfg.get("model_type", "")).strip().lower()
    if model_type == "rag":
        raise ValueError(
            "[DSTA-RAG] Invalid base_model checkpoint: config.json has model_type='rag'. "
            "Stage-2 RL requires a decoder-only causal LM checkpoint "
            "(e.g. Qwen/Llama/Mistral) or a merged SFT checkpoint. "
            f"Please update `base_model` in your stage2 config. Got: {model_path}"
        )


def build_overrides(cfg: Dict[str, Any], train_file: str, val_file: str, reward_path: str) -> List[str]:
    verl_cfg = cfg["verl"]
    reward_cfg = cfg["reward"]
    retriever = cfg["retriever"]
    overrides = [
        f"data.train_files={train_file}",
        f"data.val_files={val_file}",
        f"data.train_batch_size={verl_cfg['train_batch_size']}",
        f"data.val_batch_size={verl_cfg['val_batch_size']}",
        f"data.max_prompt_length={verl_cfg['max_prompt_length']}",
        f"data.max_response_length={verl_cfg['max_response_length']}",
        f"data.max_start_length={verl_cfg['max_start_length']}",
        f"data.max_obs_length={verl_cfg['max_obs_length']}",
        f"algorithm.adv_estimator={verl_cfg['adv_estimator']}",
        f"actor_rollout_ref.model.path={cfg['base_model']}",
        f"actor_rollout_ref.actor.optim.lr={verl_cfg['actor_lr']}",
        f"critic.optim.lr={verl_cfg['critic_lr']}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={verl_cfg['ppo_mini_batch_size']}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size={verl_cfg['ppo_micro_batch_size']}",
        f"actor_rollout_ref.rollout.name={verl_cfg['rollout_name']}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={verl_cfg['rollout_gpu_memory_utilization']}",
        f"trainer.total_epochs={verl_cfg['total_epochs']}",
        f"trainer.default_local_dir={verl_cfg['default_local_dir']}",
        f"trainer.project_name={cfg['project_name']}",
        f"trainer.experiment_name={cfg['experiment_name']}",
        f"max_turns={cfg['max_turns']}",
        f"retriever.url={retriever['url']}",
        f"retriever.topk={retriever['topk']}",
        f"++custom_reward_function.path={reward_path}",
        f"++custom_reward_function.name=compute_score",
        f"++reward.answer_weight={reward_cfg['answer_weight']}",
        f"++reward.coverage_weight={reward_cfg['coverage_weight']}",
        f"++reward.faithfulness_weight={reward_cfg['faithfulness_weight']}",
        f"++reward.stop_weight={reward_cfg['stop_weight']}",
        f"++reward.cost_weight={reward_cfg['cost_weight']}",
        f"++reward.exact_match_only={str(reward_cfg.get('exact_match_only', False)).lower()}",
        f"++reward.overconfident_stop_penalty={reward_cfg['overconfident_stop_penalty']}",
        f"++reward.underconfident_continue_penalty={reward_cfg['underconfident_continue_penalty']}",
    ]
    return overrides


def resolve_autorefine_entrypoint(
    cfg: Dict[str, Any], autorefine_root: str, run_mode: str, autorefine_cmd: str | None
) -> Path | None:
    if autorefine_cmd:
        entrypoint = Path(autorefine_cmd)
    else:
        cfg_cmd = (cfg.get("autorefine", {}).get("commands", {}) or {}).get(run_mode)
        if cfg_cmd:
            entrypoint = Path(str(cfg_cmd))
        else:
            entrypoint = Path("cmd") / f"{run_mode}.sh"

    if not entrypoint.is_absolute():
        entrypoint = Path(autorefine_root) / entrypoint

    return entrypoint if entrypoint.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch DSTA Stage-2 training on top of veRL / AutoRefine.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--autorefine-root", required=True, help="Root directory of the AutoRefine/Search-R1 style workspace.")
    parser.add_argument("--run-mode", choices=["train", "eval"], default="train", help="Whether to run AutoRefine training or evaluation entrypoint.")
    parser.add_argument(
        "--autorefine-cmd",
        default=None,
        help=(
            "AutoRefine entrypoint script path. Supports absolute path or path relative to --autorefine-root. "
            "When omitted, uses config autorefine.commands.<run-mode> or defaults to cmd/<run-mode>.sh."
        ),
    )
    parser.add_argument(
        "--no-inject-overrides",
        action="store_true",
        help="Disable hydra-style override injection. Useful when the target script manages arguments itself.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sanitize_tokenizer_config(cfg["base_model"])
    validate_model_config(cfg["base_model"])
    reward_path = str((Path(__file__).resolve().parent / "reward_fn.py").resolve())
    overrides = build_overrides(cfg, args.train_file, args.val_file, reward_path)

    entrypoint = resolve_autorefine_entrypoint(cfg, args.autorefine_root, args.run_mode, args.autorefine_cmd)
    if entrypoint is not None:
        cmd = ["bash", str(entrypoint)]
    else:
        if args.run_mode != "train":
            raise FileNotFoundError(
                f"Cannot find AutoRefine entrypoint for mode={args.run_mode}. "
                "Please provide --autorefine-cmd or set autorefine.commands in config."
            )
        cmd = ["python", "-m", cfg["verl"]["trainer_module"]]

    if not args.no_inject_overrides:
        cmd += overrides

    env = os.environ.copy()
    workspace_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = workspace_root + os.pathsep + env.get("PYTHONPATH", "")
    env["DSTA_REWARD_CFG_JSON"] = json.dumps(cfg.get("reward", {}))

    pretty = " \\\n    ".join(shlex.quote(x) for x in cmd)
    print("[DSTA-RAG] Launch command:\n" + pretty)
    print(f"[DSTA-RAG] PYTHONPATH prepended with: {workspace_root}")

    if args.dry_run:
        return

    subprocess.run(cmd, cwd=args.autorefine_root, env=env, check=True)


if __name__ == "__main__":
    main()
