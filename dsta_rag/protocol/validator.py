from __future__ import annotations

from typing import Dict, List

from .tags import VALID_ASSESS_LABELS


class ValidationError(ValueError):
    pass


def validate_protocol(parsed: Dict[str, object]) -> List[str]:
    errors: List[str] = []
    assess_label = parsed.get("assess_label")
    if assess_label is not None and assess_label not in VALID_ASSESS_LABELS:
        errors.append(f"invalid assess label: {assess_label}")

    if parsed.get("query") is None and parsed.get("answer") is None:
        errors.append("missing terminal action: neither <search> nor <answer> found")

    if parsed.get("query") and parsed.get("answer"):
        errors.append("both <search> and <answer> found in one segment")

    refine_items = parsed.get("refine_items") or []
    for idx, item in enumerate(refine_items):
        if not item.get("slot"):
            errors.append(f"refine item {idx} missing slot")
        if not item.get("claim"):
            errors.append(f"refine item {idx} missing claim")

    if parsed.get("query") and not parsed.get("next_search_target") and parsed.get("assess_label") is not None:
        errors.append("search action without <next_search_target> in <rectify>")

    return errors
