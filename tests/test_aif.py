import numpy as np
import axelrod as axl

from agents.aif.jax.aif import make_agent, ActiveInferenceAgent


def test_noisy_observation_model_at_zero_noise():
    """
    At zero noise, enabling the noisy observation model should leave the
    effective observation model A unchanged (low-level make_agent test).
    """
    # Base agent: no noisy observation model
    agent_base = make_agent(
        noise=0.0,
        use_noisy_observation_model=False,
    )

    # Test agent: noisy observation model enabled but zero noise
    agent_test = make_agent(
        noise=0.0,
        use_noisy_observation_model=True,
    )

    # A is a list of arrays; compare each corresponding element
    assert len(agent_base.A) == len(agent_test.A)

    for A_base, A_test in zip(agent_base.A, agent_test.A):
        # Use numpy for comparison since these are JAX arrays
        np.testing.assert_allclose(
            np.array(A_base),
            np.array(A_test),
            rtol=0.0,
            atol=0.0,
        )


def test_active_inference_agent_noisy_observation_model_at_zero_noise():
    """
    With the full ActiveInferenceAgent, zero noise and toggling
    use_noisy_observation_model should yield identical internal A.
    """
    agent_base = ActiveInferenceAgent(
        noise=0.0,
        use_noisy_observation_model=False,
        seed=0,
    )
    agent_test = ActiveInferenceAgent(
        noise=0.0,
        use_noisy_observation_model=True,
        seed=0,
    )

    # Underlying pymdp agent's A should be identical
    assert len(agent_base.agent.A) == len(agent_test.agent.A)
    for A_base, A_test in zip(agent_base.agent.A, agent_test.agent.A):
        np.testing.assert_allclose(
            np.array(A_base),
            np.array(A_test),
            rtol=0.0,
            atol=0.0,
        )


def _play_single_turn_match(agent):
    opponent = axl.TitForTat()
    match = axl.Match((agent, opponent), turns=1000, noise=0.0)
    match.play()
    return match.result, match.final_score()


def test_active_inference_agent_single_step_match_zero_noise():
    """
    For a one-step Axelrod match at zero noise, enabling the noisy
    observation model should not change the observed outcome.
    """
    base = ActiveInferenceAgent(
        noise=0.0,
        use_noisy_observation_model=False,
        seed=0,
        action_selection="deterministic",
    )
    test = ActiveInferenceAgent(
        noise=0.0,
        use_noisy_observation_model=True,
        seed=0,
        action_selection="deterministic",
    )

    result_base, score_base = _play_single_turn_match(base)
    result_test, score_test = _play_single_turn_match(test)

    # Observe that actions and payoffs are identical
    assert result_base == result_test
    assert score_base == score_test
