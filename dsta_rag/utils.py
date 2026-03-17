from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import orjson


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield orjson.loads(line)


def write_jsonl(items: Iterable[Dict[str, Any]], path: str | Path) -> None:
    path = ensure_parent(path)
    with Path(path).open("wb") as f:
        for item in items:
            f.write(orjson.dumps(item))
            f.write(b"\n")


def dump_json(obj: Any, path: str | Path) -> None:
    path = ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = {}
    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1
    gold_counts = {}
    for tok in gold_tokens:
        gold_counts[tok] = gold_counts.get(tok, 0) + 1
    common = 0
    for tok, c in pred_counts.items():
        common += min(c, gold_counts.get(tok, 0))
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
