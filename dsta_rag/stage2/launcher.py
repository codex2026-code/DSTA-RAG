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
        f"custom_reward_function.path={reward_path}",
        f"custom_reward_function.name=compute_score",
        f"+reward.answer_weight={reward_cfg['answer_weight']}",
        f"+reward.coverage_weight={reward_cfg['coverage_weight']}",
        f"+reward.faithfulness_weight={reward_cfg['faithfulness_weight']}",
        f"+reward.stop_weight={reward_cfg['stop_weight']}",
        f"+reward.cost_weight={reward_cfg['cost_weight']}",
        f"+reward.exact_match_only={str(reward_cfg.get('exact_match_only', False)).lower()}",
        f"+reward.overconfident_stop_penalty={reward_cfg['overconfident_stop_penalty']}",
        f"+reward.underconfident_continue_penalty={reward_cfg['underconfident_continue_penalty']}",
    ]
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch DSTA Stage-2 training on top of veRL / AutoRefine.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--autorefine-root", required=True, help="Root directory of the AutoRefine/Search-R1 style workspace.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    reward_path = str((Path(__file__).resolve().parent / "reward_fn.py").resolve())
    overrides = build_overrides(cfg, args.train_file, args.val_file, reward_path)

    cmd = ["python", "-m", cfg["verl"]["trainer_module"]] + overrides
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
