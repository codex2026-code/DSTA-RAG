from dsta_rag.protocol import parse_protocol, parse_turns


def test_parse_protocol_basic():
    text = (
        '<assess label="partial" target_slot="last_team">Bridge slot resolved.</assess>'
        '<refine><item slot="last_team" doc_id="1">David Beckham last played for Paris Saint-Germain.</item></refine>'
        '<rectify><missing_slots>home_stadium(last_team)</missing_slots><why_insufficient>Stadium unresolved.</why_insufficient><next_search_target>Paris Saint-Germain home stadium</next_search_target></rectify>'
        '<think>Search for the stadium.</think>'
        '<search>Paris Saint-Germain home stadium</search>'
    )
    parsed = parse_protocol(text)
    assert parsed["assess_label"] == "partial"
    assert parsed["refine_items"][0]["slot"] == "last_team"
    assert parsed["next_search_target"] == "Paris Saint-Germain home stadium"
    assert parsed["query"] == "Paris Saint-Germain home stadium"


def test_parse_turns_multi_turn():
    text = (
        '<think>initial</think><search>q1</search>'
        'Doc 1(Title: A) a'
        '<assess label="support" target_slot="s1">ok</assess><refine><item slot="s1" doc_id="1">c1</item></refine>'
        '<rectify><missing_slots></missing_slots><why_insufficient>done</why_insufficient><next_search_target></next_search_target></rectify>'
        '<think>done</think><answer>a1</answer>'
    )
    turns = parse_turns(text)
    assert len(turns) >= 2
    assert turns[-1]["answer"] == "a1"
