from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Dict, List

from rich.console import Console
from rich.table import Table

from dsta_rag.protocol import parse_turns
from dsta_rag.stage2.reward_fn import compute_components
from dsta_rag.utils import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay DSTA rollouts and compute task/process metrics.")
    parser.add_argument("--input", required=True, help="JSONL with {solution_str, ground_truth, extra_info} rows.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = []
    for idx, row in enumerate(read_jsonl(args.input)):
        if args.limit is not None and idx >= args.limit:
            break
        sol = row.get("solution_str") or row.get("response") or ""
        gold = row.get("ground_truth") or row.get("answer") or row.get("reward_model", {}).get("ground_truth") or ""
        extra = row.get("extra_info") or {}
        rows.append(compute_components(sol, gold, extra))

    if not rows:
        raise SystemExit("No rows found for replay evaluation.")

    table = Table(title="DSTA-RAG Replay Metrics")
    table.add_column("metric")
    table.add_column("mean")
    for key in ["answer", "coverage", "ts", "it", "ta", "faithfulness", "stop", "cost_penalty", "total"]:
        table.add_row(key, f"{mean([r[key] for r in rows]):.4f}")
    Console().print(table)


if __name__ == "__main__":
    main()
