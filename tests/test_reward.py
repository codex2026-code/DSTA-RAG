from dsta_rag.stage2.reward_fn import compute_components


def test_reward_components_smoke():
    solution = (
        '<think>start</think><search>David Beckham last professional team</search>'
        'Doc 1(Title: David Beckham) Beckham retired after Paris Saint-Germain.'
        '<assess label="partial" target_slot="last_team">team found</assess>'
        '<refine><item slot="last_team" doc_id="1">David Beckham last played for Paris Saint-Germain.</item></refine>'
        '<rectify><missing_slots>home_stadium(last_team)</missing_slots><why_insufficient>stadium missing</why_insufficient><next_search_target>Paris Saint-Germain home stadium</next_search_target></rectify>'
        '<think>continue</think><search>Paris Saint-Germain home stadium</search>'
        'Doc 1(Title: Paris Saint-Germain F.C.) PSG play at Parc des Princes.'
        '<assess label="support" target_slot="home_stadium(last_team)">stadium found</assess>'
        '<refine><item slot="home_stadium(last_team)" doc_id="1">Paris Saint-Germain plays at Parc des Princes.</item></refine>'
        '<rectify><missing_slots></missing_slots><why_insufficient>done</why_insufficient><next_search_target></next_search_target></rectify>'
        '<think>answer</think><answer>Parc des Princes</answer>'
    )
    components = compute_components(solution, 'Parc des Princes', {'oracle_slots': ['last_team', 'home_stadium(last_team)']})
    assert components['answer'] == 1.0
    assert components['coverage'] == 1.0
    assert components['faithfulness'] > 0.0
