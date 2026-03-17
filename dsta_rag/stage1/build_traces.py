from __future__ import annotations

import argparse
from typing import Dict, List, Optional

from dsta_rag.data.normalize import NormalizedExample, normalize_example
from dsta_rag.data.retrieval_cache import RetrievalCache
from dsta_rag.schema import AssessBlock, RectifyBlock, RefineItem, RetrievedDoc, Stage1Trace, TraceTurn
from dsta_rag.stage1.build_sft import convert_trace_to_sft_messages
from dsta_rag.utils import read_jsonl, write_jsonl


def docs_for_query(example: NormalizedExample, query: str, cache: Optional[RetrievalCache]) -> List[RetrievedDoc]:
    docs = []
    raw_docs = []
    if cache is not None:
        raw_docs = cache.get(example.qid, query)
    if not raw_docs:
        raw_docs = example.oracle_docs.get(query, [])
    for doc in raw_docs:
        docs.append(
            RetrievedDoc(
                doc_id=str(doc.get("doc_id") or doc.get("id") or len(docs) + 1),
                title=str(doc.get("title") or ""),
                text=str(doc.get("text") or doc.get("contents") or ""),
            )
        )
    return docs


def build_trace(example: NormalizedExample, cache: Optional[RetrievalCache]) -> Stage1Trace:
    turns: List[TraceTurn] = []
    if not example.oracle_hops:
        # fallback single-turn trace
        turns.append(
            TraceTurn(
                turn_id=1,
                query=None,
                docs=[],
                assess=None,
                refine=[],
                rectify=RectifyBlock(missing_slots=["answer"], why_insufficient="No oracle hops are available.", next_search_target=example.question),
                decision="answer",
                answer=example.answer,
            )
        )
        return Stage1Trace(example.qid, example.dataset, example.question, example.answer, turns)

    # bootstrap turn: initial search plan before first retrieval result
    first_hop = example.oracle_hops[0]
    turns.append(
        TraceTurn(
            turn_id=1,
            query=first_hop.query,
            docs=docs_for_query(example, first_hop.query, cache),
            assess=None,
            refine=[],
            rectify=RectifyBlock(
                missing_slots=[hop.slot for hop in example.oracle_hops],
                why_insufficient="No documents have been retrieved yet.",
                next_search_target=first_hop.query,
            ),
            decision="continue" if len(example.oracle_hops) > 1 else "answer",
        )
    )

    for idx, hop in enumerate(example.oracle_hops, start=2):
        is_last = idx - 1 == len(example.oracle_hops)
        remaining = [h.slot for h in example.oracle_hops[idx - 1 :]]
        refine_items = [RefineItem(slot=hop.slot, claim=hop.claim, doc_id="1", span="")]
        assess_label = "support" if is_last else "partial"
        decision = "answer" if is_last else "continue"
        query = None if is_last else example.oracle_hops[idx - 1].query
        turns.append(
            TraceTurn(
                turn_id=idx,
                query=query,
                docs=docs_for_query(example, query, cache) if query else [],
                assess=AssessBlock(
                    label=assess_label,
                    target_slot=hop.slot,
                    rationale=(
                        "The retrieved evidence resolves the current slot."
                        if is_last
                        else "The current evidence resolves an intermediate slot but additional information is still required."
                    ),
                ),
                refine=refine_items,
                rectify=RectifyBlock(
                    missing_slots=remaining if not is_last else [],
                    why_insufficient=("All slots are resolved." if is_last else "Bridge information is resolved, but downstream slots remain open."),
                    next_search_target=("" if is_last else query or ""),
                ),
                decision=decision,
                answer=(example.answer if is_last else None),
            )
        )
    return Stage1Trace(example.qid, example.dataset, example.question, example.answer, turns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DSTA stage-1 traces from raw QA data.")
    parser.add_argument("--input", required=True, help="Raw JSONL input.")
    parser.add_argument("--trace-output", required=True, help="Output JSONL for structured traces.")
    parser.add_argument("--sft-output", default=None, help="Optional output JSONL for conversational SFT data.")
    parser.add_argument("--dataset", default="auto")
    parser.add_argument("--retrieval-cache", default=None, help="Optional JSONL retrieval cache.")
    args = parser.parse_args()

    cache = RetrievalCache.from_jsonl(args.retrieval_cache) if args.retrieval_cache else None
    traces: List[Dict[str, object]] = []
    sft_rows: List[Dict[str, object]] = []
    for row in read_jsonl(args.input):
        example = normalize_example(row, dataset=args.dataset)
        trace = build_trace(example, cache)
        traces.append(trace.to_dict())
        if args.sft_output:
            sft_rows.append(convert_trace_to_sft_messages(trace))

    write_jsonl(traces, args.trace_output)
    if args.sft_output:
        write_jsonl(sft_rows, args.sft_output)


if __name__ == "__main__":
    main()
