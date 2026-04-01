from __future__ import annotations

from typing import Sequence

from .types import RubricItem, TrajectoryTurn
from .utils import maybe_truncate, normalize_whitespace


DEFAULT_PUBMED_SYSTEM_PROMPT = """You are a medical research assistant. Answer questions using available tools.

IMPORTANT:
- All reasoning, tool calls, and final answers must be written in English.
- Use tools conservatively and cite only evidence actually returned by the tools.
- For biomedical questions, prefer PubMed search before broad web search whenever feasible.
"""


CRITICAL_STEP_VOTER_SYSTEM_PROMPT = """You are a committee member identifying critical tool-use steps for Critical Step DPO.

A step is critical if model capability is the bottleneck there: a stronger expert continuation is likely to achieve a substantially better final answer than the local policy from the same state.

Focus on:
- poor search query formulation
- failure to adapt after empty or noisy evidence
- discarding or misinterpreting relevant evidence
- weak evidence synthesis or comparison logic
- wrong decision points that materially affect the final answer

Do not mark a step as critical if it is only a superficial tool hiccup with no downstream impact.

Return valid JSON only."""


RUBRIC_SCORER_SYSTEM_PROMPT = """You are an expert evaluator for biomedical research answers.

You will receive:
1. A question
2. One model answer
3. A rubric with weighted criteria

For every rubric item, assign an integer score in {0, 1, 2, 3}:
- 0: not addressed
- 1: weakly or partially addressed
- 2: mostly addressed
- 3: fully and accurately addressed

Return valid JSON only, with one score object per rubric item."""


def format_turns_for_voting(
    turns: Sequence[TrajectoryTurn],
    *,
    candidate_steps: Sequence[int] | None = None,
    max_output_chars: int = 700,
) -> str:
    """Serialize trajectory turns into a compact but judge-friendly format."""

    candidate_set = set(candidate_steps or [])
    blocks: list[str] = []
    for turn in turns:
        marker = " [CANDIDATE]" if turn.step_number in candidate_set else ""
        output_excerpt = maybe_truncate(normalize_whitespace(turn.tool_output), max_output_chars)
        thought_excerpt = maybe_truncate(normalize_whitespace(turn.think_text), 300)
        block = [
            f"Step {turn.step_number}{marker}",
            f"Tool: {turn.tool_name or 'unknown'}",
            f"Query: {turn.tool_query or ''}",
        ]
        if thought_excerpt:
            block.append(f"Reasoning: {thought_excerpt}")
        if output_excerpt:
            block.append(f"ToolOutput: {output_excerpt}")
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def build_pubmed_critical_step_prompt(
    question: str,
    turns: Sequence[TrajectoryTurn],
    *,
    candidate_steps: Sequence[int] | None = None,
    max_selected_steps: int = 2,
    rubrics: Sequence[RubricItem] | None = None,
) -> str:
    rubric_lines = ""
    if rubrics:
        rendered = [
            f"- [{item.category}] {item.title}: {item.description}"
            for item in rubrics
        ]
        rubric_lines = "Rubric Signals:\n" + "\n".join(rendered[:8]) + "\n\n"

    candidate_line = ""
    if candidate_steps:
        candidate_line = (
            "Candidate Steps: "
            + ", ".join(str(step) for step in candidate_steps)
            + "\nOnly vote on these candidates unless none of them is plausible.\n\n"
        )

    trajectory_text = format_turns_for_voting(turns, candidate_steps=candidate_steps)

    return f"""Question:
{question}

{rubric_lines}{candidate_line}Trajectory:
{trajectory_text}

Task:
Select at most {max_selected_steps} critical steps. A critical step is a capability bottleneck where a stronger expert continuation would likely produce a materially better final answer under the rubric.

Return JSON with this schema:
{{
  "critical_steps": [
    {{
      "step_number": 3,
      "is_critical": true,
      "confidence": 0.87,
      "rationale": "short explanation"
    }}
  ]
}}
"""


def build_rubric_scoring_prompt(
    question: str,
    answer: str,
    rubrics: Sequence[RubricItem],
) -> str:
    rubric_lines = []
    for idx, rubric in enumerate(rubrics, start=1):
        rubric_lines.append(
            f"{idx}. [{rubric.category}] {rubric.title} (weight={rubric.weight})\n"
            f"   {rubric.description}"
        )

    return f"""Question:
{question}

Model Answer:
{answer}

Rubric:
{chr(10).join(rubric_lines)}

Return JSON:
{{
  "scores": [
    {{
      "rubric_title": "<exact rubric title>",
      "score": 0,
      "justification": "brief explanation"
    }}
  ]
}}
"""
