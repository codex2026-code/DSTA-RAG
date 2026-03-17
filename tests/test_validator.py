from dsta_rag.protocol import parse_protocol, validate_protocol


def test_validate_protocol_ok():
    parsed = parse_protocol(
        '<assess label="miss" target_slot="s">not found</assess>'
        '<rectify><missing_slots>s</missing_slots><why_insufficient>missing</why_insufficient><next_search_target>query</next_search_target></rectify>'
        '<think>search</think><search>query</search>'
    )
    assert validate_protocol(parsed) == []


def test_validate_protocol_invalid():
    parsed = parse_protocol('<assess label="weird">x</assess><think>t</think>')
    errors = validate_protocol(parsed)
    assert errors
