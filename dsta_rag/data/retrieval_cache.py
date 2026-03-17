from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from dsta_rag.utils import read_jsonl


@dataclass
class RetrievalCache:
    by_qid_and_query: Dict[str, Dict[str, List[dict]]] = field(default_factory=dict)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "RetrievalCache":
        cache = cls()
        for row in read_jsonl(path):
            qid = str(row.get("qid") or row.get("id") or "")
            query = str(row.get("query") or "")
            docs = row.get("docs") or []
            cache.by_qid_and_query.setdefault(qid, {})[query] = docs
        return cache

    def get(self, qid: str, query: str) -> List[dict]:
        return self.by_qid_and_query.get(qid, {}).get(query, [])
