from dr_agent.cdpo.mcq_localization import localize_mcq_critical_turn


INTERLEAVED = """
<think>The answer may be propranolol because the study mentions it.</think>
<call_tool name="pubmed_search">sprague dawley albuterol tumor propranolol</call_tool>
<tool_output>Title: propranolol blocks albuterol tumorigenesis in rats</tool_output>
<think>The results feel too generic; perhaps atenolol is more likely.</think>
<call_tool name="pubmed_search">sprague dawley albuterol atenolol</call_tool>
<tool_output>Title: atenolol beta1 selectivity overview</tool_output>
""".strip()


def test_mcq_localizer_identifies_evidence_ignored_or_shift():
    result = localize_mcq_critical_turn(
        question="Which drug blocked tumorigenic effect?",
        options={
            "A": "Propranolol",
            "B": "Metoprolol",
            "C": "Atenolol",
            "D": "Carvedilol",
        },
        correct_answer="A",
        model_answer="C",
        interleaved_text=INTERLEAVED,
    )

    assert result.critical_turns
    assert result.error_subtype in {
        "misled_by_evidence",
        "evidence_ignored",
        "final_reasoning_error",
    }
