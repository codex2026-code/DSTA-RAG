from __future__ import annotations

import argparse
from typing import Dict, List

from dsta_rag.prompts import STAGE1_SYSTEM_PROMPT
from dsta_rag.schema import Stage1Trace
from dsta_rag.utils import read_jsonl, write_jsonl


def _docs_to_documents_block(docs: List[dict]) -> str:
    if not docs:
        return "<documents></documents>"
    parts = []
    for doc in docs:
        parts.append(f'Doc {doc.get("doc_id", "")}(Title: {doc.get("title", "")}) {doc.get("text", "")}')
    return f"<documents>{chr(10).join(parts)}</documents>"


def _assistant_turn(turn: dict) -> str:
    chunks: List[str] = []
    if turn.get("assess"):
        assess = turn["assess"]
        chunks.append(
            f'<assess label="{assess.get("label", "")}" target_slot="{assess.get("target_slot", "")}">{assess.get("rationale", "")}</assess>'
        )
    if turn.get("refine"):
        items = []
        for item in turn["refine"]:
            items.append(f'<item slot="{item.get("slot", "")}" doc_id="{item.get("doc_id", "")}">{item.get("claim", "")}</item>')
        chunks.append(f'<refine>{"".join(items)}</refine>')
    if turn.get("rectify"):
        rectify = turn["rectify"]
        missing = ", ".join(rectify.get("missing_slots", []))
        chunks.append(
            f'<rectify><missing_slots>{missing}</missing_slots><why_insufficient>{rectify.get("why_insufficient", "")}</why_insufficient><next_search_target>{rectify.get("next_search_target", "")}</next_search_target></rectify>'
        )
    think_text = "Update the current reasoning state and choose the next action."
    chunks.append(f'<think>{think_text}</think>')
    if turn.get("decision") == "answer":
        chunks.append(f'<answer>{turn.get("answer", "")}</answer>')
    elif turn.get("query"):
        chunks.append(f'<search>{turn.get("query", "")}</search>')
    return "".join(chunks)


def convert_trace_to_sft_messages(trace: Stage1Trace) -> Dict[str, object]:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
        {"role": "user", "content": trace.question},
    ]
    for turn in trace.turns:
        t = turn.to_dict()
        messages.append({"role": "assistant", "content": _assistant_turn(t)})
        if t.get("docs"):
            messages.append({"role": "user", "content": _docs_to_documents_block(t.get("docs", []))})
    return {
        "messages": messages,
        "metadata": {"qid": trace.qid, "dataset": trace.dataset, "answer": trace.answer},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DSTA traces to conversational SFT JSONL.")
    parser.add_argument("--input", required=True, help="Structured trace JSONL.")
    parser.add_argument("--output", required=True, help="Conversational SFT JSONL.")
    args = parser.parse_args()

    rows: List[Dict[str, object]] = []
    for row in read_jsonl(args.input):
        trace = Stage1Trace(
            qid=row["qid"],
            dataset=row["dataset"],
            question=row["question"],
            answer=row["answer"],
            turns=[],
        )
        # reuse the raw dict path instead of reconstructing dataclasses deeply
        trace_dict = row
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
            {"role": "user", "content": trace_dict["question"]},
        ]
        for turn in trace_dict["turns"]:
            messages.append({"role": "assistant", "content": _assistant_turn(turn)})
            if turn.get("docs"):
                messages.append({"role": "user", "content": _docs_to_documents_block(turn["docs"])})
        rows.append({"messages": messages, "metadata": {"qid": trace_dict["qid"], "dataset": trace_dict["dataset"], "answer": trace_dict["answer"]}})
    write_jsonl(rows, args.output)


if __name__ == "__main__":
    main()
