from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class RetrievedDoc:
    doc_id: str
    title: str
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AssessBlock:
    label: str
    target_slot: str = ""
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RefineItem:
    slot: str
    claim: str
    doc_id: str = ""
    span: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RectifyBlock:
    missing_slots: List[str] = field(default_factory=list)
    why_insufficient: str = ""
    next_search_target: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TraceTurn:
    turn_id: int
    query: Optional[str]
    docs: List[RetrievedDoc] = field(default_factory=list)
    assess: Optional[AssessBlock] = None
    refine: List[RefineItem] = field(default_factory=list)
    rectify: Optional[RectifyBlock] = None
    decision: str = "continue"
    answer: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass
class Stage1Trace:
    qid: str
    dataset: str
    question: str
    answer: str
    turns: List[TraceTurn]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qid": self.qid,
            "dataset": self.dataset,
            "question": self.question,
            "answer": self.answer,
            "turns": [t.to_dict() for t in self.turns],
        }


@dataclass
class BlackboardState:
    knowledge: List[RefineItem] = field(default_factory=list)
    missing_slots: List[str] = field(default_factory=list)
    anchors: List[str] = field(default_factory=list)
    sufficiency: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "knowledge": [item.to_dict() for item in self.knowledge],
            "missing_slots": list(self.missing_slots),
            "anchors": list(self.anchors),
            "sufficiency": float(self.sufficiency),
        }
