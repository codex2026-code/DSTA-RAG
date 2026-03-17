from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from dsta_rag.data.normalize import normalize_example
from dsta_rag.prompts import STAGE2_SYSTEM_PROMPT
from dsta_rag.utils import read_jsonl


def build_row(example: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    norm = normalize_example(example, dataset=dataset)
    oracle_slots = [hop.slot for hop in norm.oracle_hops]
    reward_meta = {
        "style": "dsta_rule",
        "ground_truth": norm.answer,
        "oracle_slots": oracle_slots,
        "oracle_hops": [hop.__dict__ for hop in norm.oracle_hops],
        "supporting_facts": example.get("supporting_facts") or [],
        "exact_match_only": False,
    }
    return {
        "data_source": norm.dataset,
        "prompt": [
            {"role": "system", "content": STAGE2_SYSTEM_PROMPT},
            {"role": "user", "content": norm.question},
        ],
        "ability": "fact-reasoning",
        "reward_model": reward_meta,
        "extra_info": {
            "split": str(example.get("split") or "train"),
            "index": str(norm.qid),
            "dataset": norm.dataset,
            "answer": norm.answer,
            "oracle_slots": oracle_slots,
            "oracle_hops": [hop.__dict__ for hop in norm.oracle_hops],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Search-R1/AutoRefine-compatible parquet for DSTA RL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="auto")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--jsonl-fallback", action="store_true", help="Also write jsonl copies when parquet engine is unavailable.")
    args = parser.parse_args()

    rows = [build_row(row, args.dataset) for row in read_jsonl(args.input)]
    if not rows:
        raise SystemExit("No rows found.")

    val_size = max(1, int(len(rows) * args.val_ratio)) if len(rows) > 1 else 1
    train_rows = rows[val_size:] if len(rows) > 1 else rows
    val_rows = rows[:val_size]

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    try:
        train_df.to_parquet(outdir / "train.parquet", index=False)
        val_df.to_parquet(outdir / "val.parquet", index=False)
    except ImportError as exc:
        if not args.jsonl_fallback:
            raise SystemExit("Parquet support requires pyarrow or fastparquet. Install with `pip install pyarrow` or rerun with --jsonl-fallback for inspection-only output.") from exc
        train_df.to_json(outdir / "train.jsonl", orient="records", lines=True, force_ascii=False)
        val_df.to_json(outdir / "val.jsonl", orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    main()
