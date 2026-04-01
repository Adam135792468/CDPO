from dr_agent.cdpo.context import (
    build_correction_context,
    build_prevention_context,
    parse_interleaved_turns,
)


INTERLEAVED = """
<think>Search broad evidence</think>
<call_tool name="pubmed_search" limit="5">drug interaction oncology phase iii</call_tool>
<tool_output>Found 0 results.</tool_output>
<think>Retry with simpler terms</think>
<call_tool name="google_search">oncology protocol exclusion anticonvulsants</call_tool>
<tool_output>Title: Trial protocol overview</tool_output>
<answer>Final answer</answer>
""".strip()


def test_parse_interleaved_turns_pairs_calls_and_outputs():
    turns = parse_interleaved_turns(INTERLEAVED)

    assert len(turns) == 2
    assert turns[0].step_number == 1
    assert turns[0].tool_name == "pubmed_search"
    assert turns[0].tool_query == "drug interaction oncology phase iii"
    assert "Found 0 results" in turns[0].tool_output

    assert turns[1].step_number == 2
    assert turns[1].tool_name == "google_search"
    assert "Retry with simpler terms" in turns[1].think_text


def test_prevention_and_correction_contexts_match_partial_rollout_contract():
    turns = parse_interleaved_turns(INTERLEAVED)

    prevention = build_prevention_context("Question text", turns, critical_step=2)
    correction = build_correction_context("Question text", turns, critical_step=2)

    # system + user + first assistant + first tool output
    assert len(prevention) == 4
    assert prevention[0]["role"] == "system"
    assert prevention[1]["role"] == "user"
    assert prevention[2]["role"] == "assistant"
    assert prevention[3]["role"] == "user"

    # correction includes second step and its tool output
    assert len(correction) == 6
    assert correction[-2]["role"] == "assistant"
    assert correction[-1]["role"] == "user"
    assert "google_search" in correction[-2]["content"]
