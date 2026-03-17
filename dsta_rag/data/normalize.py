from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OracleHop:
    slot: str
    query: str
    claim: str
    doc_title: str = ""


@dataclass
class NormalizedExample:
    qid: str
    dataset: str
    question: str
    answer: str
    oracle_hops: List[OracleHop] = field(default_factory=list)
    oracle_docs: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


def _as_doc_list(payload: Any) -> List[Dict[str, str]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        out = []
        for doc in payload:
            if isinstance(doc, dict):
                out.append(
                    {
                        "doc_id": str(doc.get("doc_id") or doc.get("id") or len(out) + 1),
                        "title": str(doc.get("title") or ""),
                        "text": str(doc.get("text") or doc.get("contents") or ""),
                    }
                )
        return out
    return []


def normalize_example(obj: Dict[str, Any], dataset: str = "auto") -> NormalizedExample:
    qid = str(obj.get("id") or obj.get("_id") or obj.get("qid") or "unknown")
    question = str(obj.get("question") or obj.get("query") or "")
    answer = str(obj.get("answer") or obj.get("gold_answer") or obj.get("solution") or "")

    oracle_hops: List[OracleHop] = []
    if isinstance(obj.get("oracle_hops"), list):
        for hop in obj["oracle_hops"]:
            if isinstance(hop, dict):
                oracle_hops.append(
                    OracleHop(
                        slot=str(hop.get("slot") or f"slot_{len(oracle_hops)+1}"),
                        query=str(hop.get("query") or hop.get("sub_question") or question),
                        claim=str(hop.get("claim") or hop.get("answer") or ""),
                        doc_title=str(hop.get("doc_title") or hop.get("title") or ""),
                    )
                )

    if not oracle_hops and isinstance(obj.get("supporting_facts"), list):
        seen_titles: List[str] = []
        for idx, fact in enumerate(obj["supporting_facts"]):
            if isinstance(fact, (list, tuple)) and fact:
                title = str(fact[0])
                if title not in seen_titles:
                    seen_titles.append(title)
                    oracle_hops.append(
                        OracleHop(
                            slot=f"hop_{len(oracle_hops)+1}",
                            query=f"{question} {title}",
                            claim=title,
                            doc_title=title,
                        )
                    )
            elif isinstance(fact, dict):
                title = str(fact.get("title") or fact.get("doc_title") or f"hop_{idx+1}")
                oracle_hops.append(
                    OracleHop(
                        slot=str(fact.get("slot") or f"hop_{len(oracle_hops)+1}"),
                        query=str(fact.get("query") or f"{question} {title}"),
                        claim=str(fact.get("claim") or title),
                        doc_title=title,
                    )
                )

    oracle_docs = {
        str(query): _as_doc_list(docs)
        for query, docs in (obj.get("oracle_docs") or {}).items()
    }

    return NormalizedExample(
        qid=qid,
        dataset=dataset,
        question=question,
        answer=answer,
        oracle_hops=oracle_hops,
        oracle_docs=oracle_docs,
        raw=obj,
    )
