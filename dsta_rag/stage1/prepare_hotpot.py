from __future__ import annotations

import argparse
from typing import Any, Dict, List, Sequence, Tuple

from dsta_rag.stage1.build_traces import build_trace
from dsta_rag.data.normalize import normalize_example
from dsta_rag.utils import write_jsonl


def _context_to_map(context: Any) -> Dict[str, List[str]]:
    """Convert HotpotQA context to {title: [sentences]} mapping."""
    title_to_sents: Dict[str, List[str]] = {}

    if isinstance(context, dict):
        titles = context.get("title") or []
        sentences = context.get("sentences") or []
        for title, sents in zip(titles, sentences):
            if isinstance(title, str) and isinstance(sents, list):
                title_to_sents[title] = [str(s).strip() for s in sents if str(s).strip()]
        return title_to_sents

    if isinstance(context, list):
        for row in context:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            title = str(row[0])
            sents_raw = row[1] if isinstance(row[1], list) else []
            title_to_sents[title] = [str(s).strip() for s in sents_raw if str(s).strip()]

    return title_to_sents


def _supporting_facts(example: Dict[str, Any]) -> List[Tuple[str, int]]:
    raw = example.get("supporting_facts")
    facts: List[Tuple[str, int]] = []

    if isinstance(raw, dict):
        titles = raw.get("title") or []
        sent_ids = raw.get("sent_id") or []
        for title, sent_id in zip(titles, sent_ids):
            try:
                idx = int(sent_id)
            except (TypeError, ValueError):
                idx = -1
            facts.append((str(title), idx))
        return facts

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    idx = int(item[1])
                except (TypeError, ValueError):
                    idx = -1
                facts.append((str(item[0]), idx))

    return facts


def hotpot_to_raw_like(example: Dict[str, Any]) -> Dict[str, Any]:
    qid = str(example.get("id") or example.get("_id") or "")
    question = str(example.get("question") or "")
    answer = str(example.get("answer") or "")
    facts = _supporting_facts(example)
    ctx_map = _context_to_map(example.get("context"))

    seen_titles: List[str] = []
    oracle_hops: List[Dict[str, str]] = []
    oracle_docs: Dict[str, List[Dict[str, str]]] = {}

    for hop_idx, (title, sent_idx) in enumerate(facts, start=1):
        if title in seen_titles:
            continue
        seen_titles.append(title)

        sentences = ctx_map.get(title, [])
        ev_text = ""
        if 0 <= sent_idx < len(sentences):
            ev_text = sentences[sent_idx]
        elif sentences:
            ev_text = " ".join(sentences)

        query = f"{question} {title}".strip()
        oracle_hops.append(
            {
                "slot": f"hop_{hop_idx}",
                "query": query,
                "claim": title,
                "doc_title": title,
            }
        )
        oracle_docs[query] = [
            {
                "doc_id": "1",
                "title": title,
                "text": ev_text,
            }
        ]

    return {
        "id": qid,
        "question": question,
        "answer": answer,
        "supporting_facts": [[t, i] for t, i in facts],
        "oracle_hops": oracle_hops,
        "oracle_docs": oracle_docs,
    }


def build_outputs(
    hf_dataset: str,
    hf_config: str,
    split: str,
    limit: int | None,
    dataset_name: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("`datasets` is required. Please install requirements-stage1.txt first.") from exc

    ds = load_dataset(hf_dataset, hf_config, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    raw_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    for row in ds:
        raw = hotpot_to_raw_like(row)
        trace = build_trace(normalize_example(raw, dataset=dataset_name), cache=None)
        raw_rows.append(raw)
        trace_rows.append(trace.to_dict())
    return raw_rows, trace_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HotpotQA from Hugging Face and convert to DSTA stage-1 formats.")
    parser.add_argument("--hf-dataset", default="hotpot_qa", help="HF dataset name.")
    parser.add_argument("--hf-config", default="fullwiki", help="HF dataset config.")
    parser.add_argument("--split", default="train", help="Dataset split to download.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of examples.")
    parser.add_argument("--dataset-name", default="hotpotqa", help="Dataset field used in traces.")
    parser.add_argument("--raw-output", default="examples/raw_hotpot_like.jsonl", help="Output JSONL path for hotpot-like rows.")
    parser.add_argument("--trace-output", default="artifacts/stage1/traces.jsonl", help="Output JSONL path for stage-1 traces.")
    args = parser.parse_args()

    raw_rows, trace_rows = build_outputs(
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        split=args.split,
        limit=args.limit,
        dataset_name=args.dataset_name,
    )
    write_jsonl(raw_rows, args.raw_output)
    write_jsonl(trace_rows, args.trace_output)


if __name__ == "__main__":
    main()
