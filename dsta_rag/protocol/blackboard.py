from __future__ import annotations

from typing import Dict

from dsta_rag.schema import BlackboardState, RefineItem


def update_blackboard(state: BlackboardState, parsed: Dict[str, object]) -> BlackboardState:
    refine_items = parsed.get("refine_items") or []
    for item in refine_items:
        state.knowledge.append(
            RefineItem(
                slot=item.get("slot", ""),
                claim=item.get("claim", ""),
                doc_id=item.get("doc_id", ""),
            )
        )
    missing = parsed.get("missing_slots") or []
    state.missing_slots = list(missing)
    target = parsed.get("next_search_target") or ""
    if target:
        state.anchors = [target]
    state.sufficiency = 1.0 if parsed.get("answer") else (0.5 if not missing else 0.0)
    return state
