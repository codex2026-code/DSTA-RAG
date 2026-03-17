from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .tags import VALID_ASSESS_LABELS


@dataclass
class ParsedTurn:
    assess_label: Optional[str] = None
    assess_target_slot: str = ""
    assess_text: str = ""
    refine_items: List[Dict[str, str]] = field(default_factory=list)
    missing_slots: List[str] = field(default_factory=list)
    why_insufficient: str = ""
    next_search_target: str = ""
    think: str = ""
    query: Optional[str] = None
    answer: Optional[str] = None


def _extract_attr(raw_tag: str, name: str) -> str:
    pattern = rf'{name}="(.*?)"'
    m = re.search(pattern, raw_tag)
    return m.group(1).strip() if m else ""


def parse_protocol(text: str) -> Dict[str, object]:
    parsed = ParsedTurn()

    assess_match = re.search(r'<assess([^>]*)>(.*?)</assess>', text, re.DOTALL)
    if assess_match:
        attrs, body = assess_match.group(1), assess_match.group(2).strip()
        label = _extract_attr(attrs, "label")
        if label and label not in VALID_ASSESS_LABELS:
            label = None
        parsed.assess_label = label
        parsed.assess_target_slot = _extract_attr(attrs, "target_slot")
        parsed.assess_text = body

    refine_match = re.search(r'<refine>(.*?)</refine>', text, re.DOTALL)
    if refine_match:
        body = refine_match.group(1)
        for item_match in re.finditer(r'<item([^>]*)>(.*?)</item>', body, re.DOTALL):
            attrs, item_body = item_match.group(1), item_match.group(2).strip()
            parsed.refine_items.append(
                {
                    "slot": _extract_attr(attrs, "slot"),
                    "doc_id": _extract_attr(attrs, "doc_id") or _extract_attr(attrs, "doc"),
                    "claim": item_body,
                }
            )

    rectify_match = re.search(r'<rectify>(.*?)</rectify>', text, re.DOTALL)
    if rectify_match:
        body = rectify_match.group(1)
        missing = re.search(r'<missing_slots>(.*?)</missing_slots>', body, re.DOTALL)
        why = re.search(r'<why_insufficient>(.*?)</why_insufficient>', body, re.DOTALL)
        next_target = re.search(r'<next_search_target>(.*?)</next_search_target>', body, re.DOTALL)
        if missing:
            raw_missing = missing.group(1).strip()
            parsed.missing_slots = [s.strip() for s in re.split(r'[\n,;]+', raw_missing) if s.strip()]
        if why:
            parsed.why_insufficient = why.group(1).strip()
        if next_target:
            parsed.next_search_target = next_target.group(1).strip()

    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        parsed.think = think_match.group(1).strip()

    search_match = re.search(r'<search>(.*?)</search>', text, re.DOTALL)
    if search_match:
        parsed.query = search_match.group(1).strip()

    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        parsed.answer = answer_match.group(1).strip()

    return {
        "assess_label": parsed.assess_label,
        "assess_target_slot": parsed.assess_target_slot,
        "assess_text": parsed.assess_text,
        "refine_items": parsed.refine_items,
        "missing_slots": parsed.missing_slots,
        "why_insufficient": parsed.why_insufficient,
        "next_search_target": parsed.next_search_target,
        "think": parsed.think,
        "query": parsed.query,
        "answer": parsed.answer,
    }


def parse_turns(text: str) -> List[Dict[str, object]]:
    """Heuristic parser for multi-turn transcripts.

    Search-R1 / AutoRefine style transcripts often interleave model actions and raw search observations.
    We therefore segment turns by the next emitted <search> or <answer> action.
    """
    spans = []
    for m in re.finditer(r'<(search|answer)>.*?</\1>', text, re.DOTALL):
        spans.append((m.start(), m.end()))
    if not spans:
        return [parse_protocol(text)]

    turns: List[Dict[str, object]] = []
    prev = 0
    for i, (_, end) in enumerate(spans):
        next_start = spans[i + 1][0] if i + 1 < len(spans) else len(text)
        segment = text[prev:next_start]
        turns.append(parse_protocol(segment))
        prev = next_start
    return turns
