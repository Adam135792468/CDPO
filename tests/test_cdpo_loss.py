import pytest


torch = pytest.importorskip("torch")

from dr_agent.cdpo.loss import rubric_cdpo_loss


def test_rubric_cdpo_loss_matches_grouped_all_pairs_shape():
    expert_policy = torch.tensor([[2.0, 1.0]], dtype=torch.float32)
    expert_ref = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    local_policy = torch.tensor([[0.2, -0.1]], dtype=torch.float32)
    local_ref = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    expert_scores = torch.tensor([[2.4, 2.1]], dtype=torch.float32)
    local_scores = torch.tensor([[0.8, 1.0]], dtype=torch.float32)
    rubric_max_scores = torch.tensor([3.0], dtype=torch.float32)
    step_weights = torch.tensor([1.25], dtype=torch.float32)

    output = rubric_cdpo_loss(
        expert_policy_logps=expert_policy,
        local_policy_logps=local_policy,
        expert_reference_logps=expert_ref,
        local_reference_logps=local_ref,
        expert_scores=expert_scores,
        local_scores=local_scores,
        rubric_max_scores=rubric_max_scores,
        step_weights=step_weights,
        beta=0.5,
        return_output=True,
    )

    assert output.loss.ndim == 0
    assert output.per_step_loss.shape == (1,)
    assert output.pair_weights.shape == (1, 2, 2)
    assert float(output.loss) > 0
