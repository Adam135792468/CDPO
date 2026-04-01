from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CDPOLossOutput:
    loss: Any
    per_step_loss: Any
    pair_weights: Any
    step_weights: Any


def rubric_cdpo_loss(
    *,
    expert_policy_logps,
    local_policy_logps,
    expert_reference_logps=None,
    local_reference_logps=None,
    expert_scores=None,
    local_scores=None,
    rubric_max_scores=None,
    step_weights=None,
    beta: float = 0.5,
    expert_mask=None,
    local_mask=None,
    reduction: str = "mean",
    return_output: bool = False,
):
    """Grouped all-pairs rubric CDPO loss from the final paper formulation.

    Shapes:
    - `expert_policy_logps`: [batch, n]
    - `local_policy_logps`: [batch, m]
    - `expert_reference_logps`: [batch, n] or None
    - `local_reference_logps`: [batch, m] or None
    - `expert_scores`: [batch, n]
    - `local_scores`: [batch, m]
    - `rubric_max_scores`: [batch]
    - `step_weights`: [batch]
    - `expert_mask`: [batch, n] bool
    - `local_mask`: [batch, m] bool
    """

    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - exercised in environments without torch
        raise ImportError("rubric_cdpo_loss requires PyTorch to be installed.") from exc

    if expert_scores is None or local_scores is None or rubric_max_scores is None:
        raise ValueError("expert_scores, local_scores, and rubric_max_scores are required.")

    if expert_reference_logps is None:
        expert_reference_logps = torch.zeros_like(expert_policy_logps)
    if local_reference_logps is None:
        local_reference_logps = torch.zeros_like(local_policy_logps)
    if step_weights is None:
        step_weights = torch.ones(
            expert_policy_logps.size(0),
            device=expert_policy_logps.device,
            dtype=expert_policy_logps.dtype,
        )
    if expert_mask is None:
        expert_mask = torch.ones_like(expert_policy_logps, dtype=torch.bool)
    if local_mask is None:
        local_mask = torch.ones_like(local_policy_logps, dtype=torch.bool)

    expert_logratios = expert_policy_logps - expert_reference_logps
    local_logratios = local_policy_logps - local_reference_logps

    logits = beta * (
        expert_logratios.unsqueeze(-1) - local_logratios.unsqueeze(-2)
    )
    pair_weights = (
        expert_scores.unsqueeze(-1) - local_scores.unsqueeze(-2)
    ) / rubric_max_scores.view(-1, 1, 1)

    pair_mask = expert_mask.unsqueeze(-1) & local_mask.unsqueeze(-2)
    log_sigmoid = F.logsigmoid(logits)
    pair_terms = -pair_weights * log_sigmoid
    pair_terms = pair_terms * pair_mask.to(pair_terms.dtype)

    pair_counts = pair_mask.sum(dim=(-1, -2)).clamp_min(1)
    per_step_loss = step_weights * pair_terms.sum(dim=(-1, -2)) / pair_counts

    if reduction == "mean":
        loss = per_step_loss.mean()
    elif reduction == "sum":
        loss = per_step_loss.sum()
    elif reduction == "none":
        loss = per_step_loss
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")

    if return_output:
        return CDPOLossOutput(
            loss=loss,
            per_step_loss=per_step_loss,
            pair_weights=pair_weights,
            step_weights=step_weights,
        )
    return loss
