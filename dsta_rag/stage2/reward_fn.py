from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from dsta_rag.protocol import parse_turns, validate_protocol
from dsta_rag.utils import normalize_text, token_f1


def _final_answer(solution_str: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    return m.group(1).strip() if m else ""


def _all_doc_ids(solution_str: str) -> set[str]:
    return set(re.findall(r"Doc\s+(\d+)", solution_str))


def _answer_score(pred: str, gold: str, exact_match_only: bool = False) -> float:
    if not pred:
        return 0.0
    if normalize_text(pred) == normalize_text(gold):
        return 1.0
    if exact_match_only:
        return 0.0
    return token_f1(pred, gold)


def _coverage_score(turns: List[Dict[str, Any]], oracle_slots: List[str]) -> float:
    if not oracle_slots:
        return 0.0
    covered = set()
    for turn in turns:
        for item in turn.get("refine_items", []):
            slot = item.get("slot")
            if slot:
                covered.add(slot)
    return len(covered.intersection(set(oracle_slots))) / max(1, len(set(oracle_slots)))


def _ts_score(turns: List[Dict[str, Any]]) -> float:
    pairs = []
    for i in range(len(turns) - 1):
        target = normalize_text(str(turns[i].get("next_search_target") or ""))
        next_query = normalize_text(str(turns[i + 1].get("query") or ""))
        if not target and not next_query:
            continue
        if not target or not next_query:
            pairs.append(0.0)
        elif target == next_query:
            pairs.append(1.0)
        else:
            pairs.append(token_f1(target, next_query))
    return sum(pairs) / len(pairs) if pairs else 0.0


def _it_score(turns: List[Dict[str, Any]], solution_str: str) -> float:
    valid_doc_ids = _all_doc_ids(solution_str)
    scores = []
    for turn in turns:
        items = turn.get("refine_items", []) or []
        if not items:
            continue
        grounded = 0
        for item in items:
            doc_id = str(item.get("doc_id") or "")
            claim = str(item.get("claim") or "")
            if claim and (not doc_id or doc_id in valid_doc_ids):
                grounded += 1
        scores.append(grounded / len(items))
    return sum(scores) / len(scores) if scores else 0.0


def _ta_score(pred: str, turns: List[Dict[str, Any]]) -> float:
    pred_norm = normalize_text(pred)
    if not pred_norm:
        return 0.0
    claims = " ".join(item.get("claim", "") for turn in turns for item in (turn.get("refine_items") or []))
    claims_norm = normalize_text(claims)
    if not claims_norm:
        return 0.0
    return 1.0 if pred_norm in claims_norm or claims_norm in pred_norm else token_f1(pred_norm, claims_norm)


def _stop_score(pred: str, turns: List[Dict[str, Any]], answer_score: float, coverage_score: float, over_pen: float, under_pen: float) -> float:
    has_answer = bool(pred)
    if has_answer and coverage_score >= 0.999 and answer_score >= 0.999:
        return 1.0
    if has_answer and coverage_score < 0.999:
        return -float(over_pen)
    if (not has_answer) and coverage_score >= 0.999:
        return -float(under_pen)
    return 0.0


def _cost_penalty(solution_str: str) -> float:
    search_calls = len(re.findall(r"<search>", solution_str))
    token_est = max(1, len(solution_str.split()))
    return 0.01 * search_calls + 0.0001 * token_est


def _env_reward_cfg() -> Dict[str, Any]:
    raw = os.environ.get("DSTA_REWARD_CFG_JSON", "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def compute_components(solution_str: str, ground_truth: str, extra_info: Dict[str, Any] | None = None) -> Dict[str, float]:
    extra_info = extra_info or {}
    cfg = extra_info.get("reward_cfg", {}) or _env_reward_cfg()
    exact_match_only = bool(cfg.get("exact_match_only", extra_info.get("exact_match_only", False)))
    oracle_slots = list(extra_info.get("oracle_slots") or [])
    over_pen = float(cfg.get("overconfident_stop_penalty", extra_info.get("overconfident_stop_penalty", 1.0)))
    under_pen = float(cfg.get("underconfident_continue_penalty", extra_info.get("underconfident_continue_penalty", 0.25)))
    answer_weight = float(cfg.get("answer_weight", 1.0))
    coverage_weight = float(cfg.get("coverage_weight", 0.25))
    faithfulness_weight = float(cfg.get("faithfulness_weight", 0.25))
    stop_weight = float(cfg.get("stop_weight", 0.15))
    cost_weight = float(cfg.get("cost_weight", 0.02))

    turns = parse_turns(solution_str)
    pred = _final_answer(solution_str)

    answer_score = _answer_score(pred, ground_truth, exact_match_only=exact_match_only)
    coverage_score = _coverage_score(turns, oracle_slots)
    ts_score = _ts_score(turns)
    it_score = _it_score(turns, solution_str)
    ta_score = _ta_score(pred, turns)
    faithfulness = (ts_score + it_score + ta_score) / 3.0
    stop_score = _stop_score(pred, turns, answer_score, coverage_score, over_pen, under_pen)
    cost_penalty = _cost_penalty(solution_str)

    # schema validity helps suppress malformed protocol padding
    validation_penalty = 0.0
    for turn in turns:
        errs = validate_protocol(turn)
        validation_penalty += 0.05 * len(errs)

    total = (
        answer_weight * answer_score
        + coverage_weight * coverage_score
        + faithfulness_weight * faithfulness
        + stop_weight * stop_score
        - cost_weight * cost_penalty
        - validation_penalty
    )

    return {
        "answer": float(answer_score),
        "coverage": float(coverage_score),
        "ts": float(ts_score),
        "it": float(it_score),
        "ta": float(ta_score),
        "faithfulness": float(faithfulness),
        "stop": float(stop_score),
        "cost_penalty": float(cost_penalty),
        "validation_penalty": float(validation_penalty),
        "total": float(total),
    }


def compute_score(data_source: str, solution_str: str, ground_truth: Any, extra_info: Dict[str, Any] | None = None) -> float:
    gold = ground_truth if isinstance(ground_truth, str) else str(ground_truth)
    components = compute_components(solution_str, gold, extra_info=extra_info)
    return float(components["total"])
