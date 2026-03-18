from dsta_rag.stage2.build_rl_dataset import build_row


def test_build_row_normalizes_supporting_facts_for_parquet() -> None:
    example = {
        "_id": "q1",
        "question": "Who wrote Hamlet?",
        "answer": "William Shakespeare",
        "supporting_facts": [["Hamlet", 0], {"title": "William Shakespeare", "sent_id": 3}, "misc"],
        "context": [["Hamlet", ["Hamlet is a tragedy written by William Shakespeare."]]],
    }

    row = build_row(example, dataset="hotpotqa")
    supporting_facts = row["reward_model"]["supporting_facts"]

    assert supporting_facts == [
        {"title": "Hamlet", "sent_id": 0},
        {"title": "William Shakespeare", "sent_id": 3},
        {"title": "misc", "sent_id": None},
    ]
