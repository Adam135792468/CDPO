from __future__ import annotations

import re
from typing import Iterable, Sequence

from .prompts import DEFAULT_PUBMED_SYSTEM_PROMPT
from .types import PartialRolloutRecord, RubricItem, TrajectoryTurn


CALL_TOOL_PATTERN = re.compile(
    r'<call_tool\s+name="([^"]+)"(?:\s+([^>]*))?>(.*?)</call_tool>',
    re.DOTALL,
)
TOOL_OUTPUT_PATTERN = re.compile(r"<tool_output>.*?</tool_output>", re.DOTALL)
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
PARAM_PATTERN = re.compile(r'(\w+)="([^"]*)"')


def _parse_params(params_str: str | None) -> dict[str, str]:
    if not params_str:
        return {}
    return {match.group(1): match.group(2) for match in PARAM_PATTERN.finditer(params_str)}


def parse_interleaved_turns(interleaved_text: str) -> list[TrajectoryTurn]:
    """Parse tool-augmented trajectory text into ordered tool-call turns.

    The current training data stores one assistant/tool interaction loop as:
    `<think>...</think><call_tool ...>...</call_tool><tool_output>...</tool_output>`.
    This parser keeps the assistant content and the following tool output paired
    together so that critical-step selection and context export share the same
    turn representation.
    """

    if not interleaved_text:
        return []

    call_matches = list(CALL_TOOL_PATTERN.finditer(interleaved_text))
    output_matches = list(TOOL_OUTPUT_PATTERN.finditer(interleaved_text))

    turns: list[TrajectoryTurn] = []
    output_index = 0
    previous_boundary = 0

    for turn_index, call_match in enumerate(call_matches, start=1):
        next_call_start = (
            call_matches[turn_index].start()
            if turn_index < len(call_matches)
            else len(interleaved_text)
        )

        tool_name = call_match.group(1)
        params = _parse_params(call_match.group(2))
        tool_query = call_match.group(3).strip()

        assistant_start = previous_boundary
        assistant_end = call_match.end()
        assistant_content = interleaved_text[assistant_start:assistant_end].strip()

        paired_output = ""
        while output_index < len(output_matches):
            output_match = output_matches[output_index]
            if output_match.start() < call_match.end():
                output_index += 1
                continue
            if output_match.start() >= next_call_start:
                break
            paired_output = output_match.group(0).strip()
            previous_boundary = output_match.end()
            output_index += 1
            break
        else:
            previous_boundary = call_match.end()

        think_match = THINK_PATTERN.search(assistant_content)
        think_text = think_match.group(1).strip() if think_match else ""

        turns.append(
            TrajectoryTurn(
                step_number=turn_index,
                assistant_content=assistant_content,
                tool_output=paired_output,
                think_text=think_text,
                tool_name=tool_name,
                tool_query=tool_query,
                tool_parameters=params,
            )
        )

        if previous_boundary < call_match.end():
            previous_boundary = call_match.end()

    return turns


def _base_messages(question: str, system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def build_prevention_context(
    question: str,
    turns: Sequence[TrajectoryTurn],
    critical_step: int,
    *,
    system_prompt: str = DEFAULT_PUBMED_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """Build the proactive context truncated before the critical step."""

    messages = _base_messages(question, system_prompt)
    for turn in turns:
        if turn.step_number >= critical_step:
            break
        messages.append({"role": "assistant", "content": turn.assistant_content})
        if turn.tool_output:
            messages.append({"role": "user", "content": turn.tool_output})
    return messages


def build_correction_context(
    question: str,
    turns: Sequence[TrajectoryTurn],
    critical_step: int,
    *,
    system_prompt: str = DEFAULT_PUBMED_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """Build the post-hoc correction context including the critical step output."""

    messages = _base_messages(question, system_prompt)
    for turn in turns:
        if turn.step_number > critical_step:
            break
        messages.append({"role": "assistant", "content": turn.assistant_content})
        if turn.tool_output:
            messages.append({"role": "user", "content": turn.tool_output})
    return messages


def build_partial_context_records(
    sample: dict,
    critical_steps: Iterable[int],
    *,
    system_prompt: str = DEFAULT_PUBMED_SYSTEM_PROMPT,
) -> list[dict]:
    """Export prevention/correction context records compatible with partial rollout JSONL."""

    turns = parse_interleaved_turns(str(sample.get("interleaved_text", "")))
    if not turns:
        return []

    max_step = len(turns)
    unique_steps = []
    seen: set[int] = set()
    for step in critical_steps:
        if step in seen:
            continue
        seen.add(step)
        if 1 <= step <= max_step:
            unique_steps.append(step)

    all_rubrics = sample.get("all_rubrics") or []
    tool_rubrics = sample.get("tool_rubrics") or []
    content_rubrics = sample.get("content_rubrics") or []

    outputs: list[dict] = []
    for step in unique_steps:
        prevention = build_prevention_context(
            str(sample.get("question", "")),
            turns,
            step,
            system_prompt=system_prompt,
        )
        correction = build_correction_context(
            str(sample.get("question", "")),
            turns,
            step,
            system_prompt=system_prompt,
        )
        common = {
            "sample_id": sample.get("sample_id", ""),
            "question": sample.get("question", ""),
            "topic": sample.get("topic", ""),
            "question_type": sample.get("question_type", ""),
            "critical_step": step,
            "total_original_steps": len(turns),
            "tool_rubrics": tool_rubrics,
            "content_rubrics": content_rubrics,
            "all_rubrics": all_rubrics,
            "dr_tulu_results": [],
            "openrouter_results": [],
        }
        outputs.append(
            {
                **common,
                "context_type": "prevention",
                "context_messages": prevention,
            }
        )
        outputs.append(
            {
                **common,
                "context_type": "correction",
                "context_messages": correction,
            }
        )
    return outputs
