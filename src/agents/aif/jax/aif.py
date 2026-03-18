import jax.numpy as jnp
import jax.tree_util as jtu
from jax import nn, vmap, lax, jit
from jax import random as jr
import numpy as np
from collections import deque
import warnings

from pymdp.envs import GridWorldEnv
from pymdp.jax.task import PyMDPEnv
from pymdp.jax.agent import Agent as AIFAgent

import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional, Tuple

import axelrod
import numpy as np
from axelrod.action import Action
from pymdp.agent import Agent
import pymdp.utils as utils
from ...wrapper import C, JointWrapper

# Constants
START = 0
CC = 1
CD = 2
DC = 3
DD = 4

C = 0
D = 1
ALL_STATES = [START, CC, CD, DC, DD]
ALL_STATES_LABELS = ["START", "CC", "CD", "DC", "DD"]
ALL_OBS = [START, CC, CD, DC, DD]
ALL_OBS_LABELS = ["START", "CC", "CD", "DC", "DD"]
ALL_ACTIONS = [C, D]
ALL_ACTIONS_LABELS = ["C", "D"]
n_batches = 1
num_states = len(ALL_STATES)
num_obs = len(ALL_OBS)
num_actions = len(ALL_ACTIONS)


def noisy_obs_matrix(n: float, P0: np.ndarray | None = None) -> np.ndarray:
    """
    Returns P_noisy(o|s) given noise n and a base emission matrix P0(o|s).
    Convention: rows=observations, cols=true states; each column sums to 1.
    If P0 is None, uses the identity (no base noise).
    """
    S = np.array([[1 - n, n],
                  [n, 1 - n]], dtype=float)        # single-player BSC(n)
    K = np.kron(S, S)                               # two players: independent noise
    if P0 is None:
        return K
    return K @ P0                                   # compose channels


def make_agent(
    lr_B: float = 1,
    policy_len: int = 10,
    alpha: float = 1,
    gamma: float = 1,
    bias: float = 0.5,
    cooperative_preference: bool = False,
    pB_scale: float = 100,
    noise: float = 0.0,
    use_noisy_observation_model: bool = False,
    action_selection: str = "stochastic",
    use_utility: bool = True,
    use_states_info_gain: bool = True,
    use_param_info_gain: bool = True,
    preference_params: Tuple[float, float, float, float] = (3.0, 1.0, 0.0, 5.0),
):
    """
    Create an Active Inference agent with configurable parameters.
    
    Args:
        lr_B: Learning rate for B matrix
        policy_len: Policy horizon length
        alpha: Action selection precision
        gamma: Policy precision
        bias: Bias for state transitions
        cooperative_preference: If True, use cooperative (nash) preferences; if False, use standard preferences
        pB_scale: Scale for Dirichlet prior over B
        noise: Noise level for observations (0.0 = no noise)
        use_noisy_observation_model: If True, apply noise to A matrix; if False, use perfect observations
        action_selection: "stochastic" or "deterministic"
        use_utility: Whether to use utility (preference) term
        use_states_info_gain: Whether to use state information gain
        use_param_info_gain: Whether to use parameter information gain
        preference_params: Tuple of (R, P, S, T) payoffs
    """
    A_1 = np.eye(num_states)
    
    # Apply noise to observation matrix if requested and noise > 0
    if use_noisy_observation_model and noise > 0.0:
        P0 = np.eye(num_states - 1)
        P_noisy = noisy_obs_matrix(noise, P0)
        A_1[1:, 1:] = P_noisy
        assert A_1.sum(axis=0, keepdims=True).all() == 1

    B_1 = np.zeros((num_states, num_states, num_actions))
    # if I cooperate in any state, The next state is CC or CD with equal probability
    B_1[CC, :, C] = bias
    B_1[CD, :, C] = 1 - bias
    # if I defect in any state, The next state is DC or DD with equal probability
    B_1[DC, :, D] = 1 - bias
    B_1[DD, :, D] = bias

    C_1 = np.zeros(num_obs)
    R, P, S, T = preference_params
    
    if cooperative_preference:
        # Nash/cooperative preferences: no temptation to defect
        C_1[CC] = R
        C_1[CD] = S
        C_1[DC] = S  # Remove temptation
        C_1[DD] = P
    else:
        # Standard preferences: temptation to defect for higher payoff
        C_1[CC] = R
        C_1[CD] = S
        C_1[DC] = T
        C_1[DD] = P

    D_1 = np.zeros(num_states)
    D_1[START] = 1

    FINAL_A = [jnp.broadcast_to(A_1, (n_batches,) + (num_obs, num_states))]
    FINAL_B = [jnp.broadcast_to(B_1, (n_batches,) + (num_states, num_states, num_actions))]
    FINAL_C = [jnp.broadcast_to(C_1, (n_batches,) + (num_obs,))]
    FINAL_D = [jnp.broadcast_to(D_1, (n_batches,) + (num_states,))]

    # pb would be a dirichlet distribution of FINAL_B[0] scaled by pB_scale
    # Suppress the pymdp warning about object array casting
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Input array is not an object array", module="pymdp.utils")
        pB = jnp.array(utils.dirichlet_like(FINAL_B[0], scale=pB_scale)[0])
    pB = [jnp.broadcast_to(pB, (n_batches,) + (num_states, num_states, num_actions))]

    agent = AIFAgent(
        FINAL_A,
        FINAL_B,
        FINAL_C,
        FINAL_D,
        E=None,
        pA=None,
        pB=pB,
        policy_len=policy_len,
        use_utility=use_utility,
        use_states_info_gain=use_states_info_gain,
        use_param_info_gain=use_param_info_gain,
        gamma=jnp.ones(1) * gamma,
        alpha=jnp.ones(1) * alpha,
        onehot_obs=False,
        action_selection=action_selection,
        inference_algo="ovf",
        num_iter=1,
        learn_A=False,
        learn_B=True,
        learn_D=False,
        sampling_mode="marginal",
    )

    return agent


def action_pair_to_obs(action, opponent_action):
    own_action = action
    opponent_action = opponent_action
    # CC -> 1 , CD -> 2, DC -> 3, DD -> 4
    # get idx from pair
    idx = 1 + (own_action * 2) + opponent_action
    return jnp.array([[idx]])


@jit
def step(rng_key, agent, obs_idx, empirical_prior):
    qs = agent.infer_states(obs_idx, empirical_prior)
    qpi, _ = agent.infer_policies(qs)
    rng_key = jr.split(rng_key)
    action = agent.sample_action(qpi, rng_key=rng_key[1:])
    empirical_prior, qs = agent.update_empirical_prior(action, qs)
    return rng_key[0], obs_idx, empirical_prior, qs, action


@jit
def step_with_diagnostics(rng_key, agent, obs_idx, empirical_prior):
    qs = agent.infer_states(obs_idx, empirical_prior)
    qpi, G, diagnostics = agent.infer_policies(qs, return_diagnostics=True)
    rng_key = jr.split(rng_key)
    action = agent.sample_action(qpi, rng_key=rng_key[1:])
    empirical_prior, qs = agent.update_empirical_prior(action, qs)
    return rng_key[0], obs_idx, empirical_prior, qs, action, qpi, G, diagnostics


@jit
def update_B(agent, beliefs, outcomes, actions, lr_B=1, pB_decay_rate=1.0):
    agent = agent.infer_parameters(beliefs, outcomes, actions, lr_B=lr_B)
    # Always apply decay (no-op when pB_decay_rate == 1.0)
    agent.pB[0] = agent.pB[0] * pB_decay_rate
    return agent


class ActiveInferenceAgent(JointWrapper):
    name = "ActiveInferenceAgent"

    def __init__(
        self,
        policy_len: int = 5,
        update_interval: int = 10,
        seed: int = 0,
        lr_B: float = 1,
        alpha: float = 1,
        gamma: float = 1,
        bias: float = 0.5,
        cooperative_preference: bool = False,
        pB_scale: float = 1,
        action_selection: str = "deterministic",
        use_utility: bool = True,
        use_states_info_gain: bool = True,
        use_param_info_gain: bool = True,
        noise: float = 0.0,
        use_noisy_observation_model: bool = False,
        pB_decay_rate: float = 1.0,
    ) -> None:
        super().__init__()
        # Store all configuration parameters
        self.policy_len = policy_len
        self.update_interval = update_interval
        self.seed = seed
        self.lr_B = lr_B
        self.alpha = alpha
        self.gamma = gamma
        self.bias = bias
        self.cooperative_preference = cooperative_preference
        self.pB_scale = pB_scale
        self.action_selection = action_selection
        self.use_utility = use_utility
        self.use_states_info_gain = use_states_info_gain
        self.use_param_info_gain = use_param_info_gain
        self.noise = noise
        self.use_noisy_observation_model = use_noisy_observation_model
        self.pB_decay_rate = pB_decay_rate
        self.preference_params = (3.0, 1.0, 0.0, 5.0)
        
        # Use deques with appropriate maxlen for rolling window
        # For N transitions: need N+1 beliefs/obs, N+1 actions (we slice off last)
        self.qs = deque(maxlen=self.update_interval + 1)
        self.obs = deque(maxlen=self.update_interval + 1)
        self.actions = deque(maxlen=self.update_interval + 1)
        self.qpi_history = []  # Store policy distributions
        self.efe_history = []  # Store EFE values
        self.step_count = 0  # Track steps since last update
        
        # Create agent and initialize
        self.agent = self.make_agent()
        self.empirical_prior = self.agent.D
        self.set_seed(seed)
    
    def make_agent(self):
        """Create agent with current configuration parameters."""
        return make_agent(
            lr_B=self.lr_B,
            policy_len=self.policy_len,
            alpha=self.alpha,
            gamma=self.gamma,
            bias=self.bias,
            cooperative_preference=self.cooperative_preference,
            pB_scale=self.pB_scale,
            noise=self.noise,
            use_noisy_observation_model=self.use_noisy_observation_model,
            action_selection=self.action_selection,
            use_utility=self.use_utility,
            use_states_info_gain=self.use_states_info_gain,
            use_param_info_gain=self.use_param_info_gain,
            preference_params=self.preference_params,
        )
    
    def receive_match_attributes(self) -> None:  # type: ignore[override]
        """Recreate agent when match attributes are received (e.g., noise level)."""
        super().receive_match_attributes()
        if "game" in self.match_attributes:
            self.preference_params = self.match_attributes["game"].RPST()
        # Recreate agent to handle any updated attributes like noise
        if hasattr(self, 'agent') and self.agent is not None:
            self.agent = self.make_agent()

    def set_seed(self, seed: int):
        self.seed = seed
        self.rng_key = jr.PRNGKey(self.seed)
        self.rng_key = jr.split(self.rng_key)[1]

    def step(self, state: Tuple[Optional[Action], Optional[Action]]) -> Action:
        obs_idx = None
        if state[0] is None and state[1] is None:
            obs_idx = [jnp.broadcast_to(jnp.array([START]), (1, 1))]
        else:
            obs_idx = [action_pair_to_obs(state[0].value, state[1].value)]
        
        self.rng_key = jr.split(self.rng_key)[1]
        rng_key, obs_idx, empirical_prior, qs, action = step(
            self.rng_key, self.agent, obs_idx, self.empirical_prior
        )
        self.empirical_prior = empirical_prior
        
        # Append in original order: qs, obs, actions together
        # action[t] is taken from qs[t], and results in obs[t+1], qs[t+1] 
        self.qs.append(qs)
        self.obs.append(obs_idx)
        self.actions.append(action)
        self.step_count += 1
        
        # Update every N steps once we have enough data
        # We need N+1 observations/beliefs but only N actions (exclude the last action that hasn't been executed yet)
        if self.step_count >= self.update_interval and len(self.actions) >= self.update_interval and len(self.qs) >= self.update_interval + 1:
            # Convert deques to arrays
            qs_arr = jnp.array([q for q in self.qs])
            obs_arr = jnp.array([o for o in self.obs])
            act_arr = jnp.array([a for a in self.actions])
            
            # Reshape for batch processing
            beliefs = [qs_arr.reshape(n_batches, len(self.qs), num_states)]
            outcomes = obs_arr.reshape(n_batches, len(self.obs))
            # Exclude last action - we haven't seen its result yet
            actions = act_arr[:-1].reshape(n_batches, len(self.actions) - 1, 1)
            
            self.agent = update_B(
                self.agent, beliefs, outcomes, actions, 
                lr_B=self.lr_B, pB_decay_rate=self.pB_decay_rate
            )
            
            self.qpi_history = []
            self.efe_history = []
            self.step_count = 0  # Reset counter after update
        
        return Action.C if action[0][0] == C else Action.D

    def reset(self) -> None:  
        self.qs = deque(maxlen=self.update_interval + 1)
        self.obs = deque(maxlen=self.update_interval + 1)
        self.actions = deque(maxlen=self.update_interval + 1)
        self.qpi_history = []
        self.efe_history = []
        self.step_count = 0
        self.empirical_prior = self.agent.D
        super().reset()


class DiagnosticAIFAgent(ActiveInferenceAgent):
    """ActiveInferenceAgent that captures per-timestep EFE decomposition.

    At every step the agent records, for each possible first action (C or D),
    the q_pi-weighted EFE components summed across the planning horizon:
      - utility          (expected preference satisfaction)
      - state_info_gain  (state information gain / salience)
      - param_info_gain  (parameter information gain / novelty, B matrix only)
      - total_neg_efe    (total negative EFE = sum of above)

    Results are stored in ``self.efe_decomposition_history`` as a list of dicts,
    one dict per timestep.
    """

    name = "DiagnosticAIFAgent"

    def __init__(
        self,
        policy_len: int = 5,
        update_interval: int = 10,
        seed: int = 0,
        lr_B: float = 1,
        alpha: float = 1,
        gamma: float = 1,
        bias: float = 0.5,
        cooperative_preference: bool = False,
        pB_scale: float = 1,
        action_selection: str = "deterministic",
        use_utility: bool = True,
        use_states_info_gain: bool = True,
        use_param_info_gain: bool = True,
        noise: float = 0.0,
        use_noisy_observation_model: bool = False,
        pB_decay_rate: float = 1.0,
    ) -> None:
        super().__init__(
            policy_len=policy_len,
            update_interval=update_interval,
            seed=seed,
            lr_B=lr_B,
            alpha=alpha,
            gamma=gamma,
            bias=bias,
            cooperative_preference=cooperative_preference,
            pB_scale=pB_scale,
            action_selection=action_selection,
            use_utility=use_utility,
            use_states_info_gain=use_states_info_gain,
            use_param_info_gain=use_param_info_gain,
            noise=noise,
            use_noisy_observation_model=use_noisy_observation_model,
            pB_decay_rate=pB_decay_rate,
        )
        self.efe_decomposition_history = []

    def _extract_efe_decomposition(self, qpi, G, diagnostics):
        """Summarise per-policy diagnostics into per-first-action EFE components.

        The library computes:
            step_neg_G = info_gain + utility
                         - param_info_gain_a - param_info_gain_b
                         + inductive_value

        We store each term's *contribution* to neg_G so that:
            utility + state_info_gain + param_info_gain == total_neg_efe
        This means param_info_gain is stored as ``-(raw_pB + raw_pA)``
        (the actual contribution to neg_G, not the raw KL value).

        Parameters
        ----------
        qpi : jnp.ndarray, shape ``(batch, n_policies)``
            Posterior over policies.
        G : jnp.ndarray, shape ``(batch, n_policies)``
            Neg-EFE per policy returned by ``infer_policies``.
        diagnostics : dict
            Per-policy, per-planning-step diagnostics from the library.
            Scalar fields have shape ``(batch, n_policies, policy_len)``.
        """
        policies = self.agent.policies          # (n_policies, policy_len, n_factors)
        n_policies = policies.shape[0]
        pol_len    = policies.shape[1]

        first_actions = policies[:, 0, 0]       # (n_policies,)

        # --- shape assertions on diagnostics ---------------------------------
        for key in ("utility", "info_gain", "param_info_gain_a",
                    "param_info_gain_b", "step_neg_G"):
            val = diagnostics[key]
            assert val.shape == (1, n_policies, pol_len), (
                f"diagnostics['{key}'] expected shape "
                f"(1, {n_policies}, {pol_len}), got {val.shape}"
            )

        assert qpi.shape == (1, n_policies), (
            f"qpi expected (1, {n_policies}), got {qpi.shape}"
        )
        assert G.shape == (1, n_policies), (
            f"G expected (1, {n_policies}), got {G.shape}"
        )

        # --- assert clean partition: every policy starts with C or D ---------
        assert jnp.all((first_actions == C) | (first_actions == D)), (
            "Not all policies start with C(0) or D(1)"
        )
        n_c = int((first_actions == C).sum())
        n_d = int((first_actions == D).sum())
        assert n_c + n_d == n_policies, (
            f"Partition error: {n_c} + {n_d} != {n_policies}"
        )

        # --- sum each diagnostic component across the planning horizon -------
        # (batch=0, n_policies, policy_len) -> (n_policies,)
        utility_per_pol     = diagnostics["utility"][0].sum(axis=-1)
        info_gain_per_pol   = diagnostics["info_gain"][0].sum(axis=-1)
        raw_pA_per_pol      = diagnostics["param_info_gain_a"][0].sum(axis=-1)
        raw_pB_per_pol      = diagnostics["param_info_gain_b"][0].sum(axis=-1)
        step_G_per_pol      = diagnostics["step_neg_G"][0].sum(axis=-1)

        # The *contribution* of param info gain to neg_G is -(pA + pB).
        param_contrib_per_pol = -(raw_pA_per_pol + raw_pB_per_pol)

        # --- cross-check: sum of step_neg_G == G from infer_policies ---------
        G_batch0 = G[0]                         # (n_policies,)
        assert jnp.allclose(step_G_per_pol, G_batch0, atol=1e-4), (
            "sum(step_neg_G, horizon) != G  "
            f"max diff = {float(jnp.max(jnp.abs(step_G_per_pol - G_batch0)))}"
        )

        # --- cross-check: components add up to step_neg_G -------------------
        reconstructed = info_gain_per_pol + utility_per_pol + param_contrib_per_pol
        assert jnp.allclose(reconstructed, step_G_per_pol, atol=1e-4), (
            "utility + info_gain - (pA + pB) != step_neg_G  "
            f"max diff = {float(jnp.max(jnp.abs(reconstructed - step_G_per_pol)))}"
        )

        # --- aggregate by first action (q_pi-weighted mean) -----------------
        q_pi = qpi[0]                           # (n_policies,)

        result = {}
        for action_idx, action_label in ((C, "C"), (D, "D")):
            mask = (first_actions == action_idx)
            masked_q = jnp.where(mask, q_pi, 0.0)
            total_q  = masked_q.sum()
            safe_total_q = jnp.where(total_q > 0, total_q, 1.0)

            def weighted_mean(vals, _mq=masked_q, _sq=safe_total_q):
                return (_mq * vals).sum() / _sq

            u  = float(weighted_mean(utility_per_pol))
            si = float(weighted_mean(info_gain_per_pol))
            pi = float(weighted_mean(param_contrib_per_pol))
            total = float(weighted_mean(step_G_per_pol))

            assert abs(u + si + pi - total) < 1e-4, (
                f"action={action_label}: u({u}) + si({si}) + pi({pi}) "
                f"= {u + si + pi} != total({total})"
            )

            result[action_label] = {
                "utility":         u,
                "state_info_gain": si,
                "param_info_gain": pi,
                "total_neg_efe":   total,
            }

        # --- marginal action probabilities -----------------------------------
        c_mask = (first_actions == C)
        d_mask = (first_actions == D)
        q_pi_C = float(jnp.where(c_mask, q_pi, 0.0).sum())
        q_pi_D = float(jnp.where(d_mask, q_pi, 0.0).sum())

        assert abs(q_pi_C + q_pi_D - 1.0) < 1e-4, (
            f"q_pi_C({q_pi_C}) + q_pi_D({q_pi_D}) = {q_pi_C + q_pi_D} != 1.0"
        )

        result["q_pi_C"] = q_pi_C
        result["q_pi_D"] = q_pi_D
        return result

    def step(self, state: Tuple[Optional[Action], Optional[Action]]) -> Action:
        obs_idx = None
        if state[0] is None and state[1] is None:
            obs_idx = [jnp.broadcast_to(jnp.array([START]), (1, 1))]
        else:
            obs_idx = [action_pair_to_obs(state[0].value, state[1].value)]

        self.rng_key = jr.split(self.rng_key)[1]
        rng_key, obs_idx, empirical_prior, qs, action, qpi, G, diagnostics = step_with_diagnostics(
            self.rng_key, self.agent, obs_idx, self.empirical_prior
        )
        self.empirical_prior = empirical_prior

        # Record EFE decomposition for this timestep
        decomp = self._extract_efe_decomposition(qpi, G, diagnostics)
        self.efe_decomposition_history.append(decomp)

        self.qs.append(qs)
        self.obs.append(obs_idx)
        self.actions.append(action)
        self.step_count += 1

        if self.step_count >= self.update_interval and len(self.actions) >= self.update_interval and len(self.qs) >= self.update_interval + 1:
            qs_arr = jnp.array([q for q in self.qs])
            obs_arr = jnp.array([o for o in self.obs])
            act_arr = jnp.array([a for a in self.actions])

            beliefs = [qs_arr.reshape(n_batches, len(self.qs), num_states)]
            outcomes = obs_arr.reshape(n_batches, len(self.obs))
            actions = act_arr[:-1].reshape(n_batches, len(self.actions) - 1, 1)

            self.agent = update_B(
                self.agent, beliefs, outcomes, actions,
                lr_B=self.lr_B, pB_decay_rate=self.pB_decay_rate
            )
            self.step_count = 0

        return Action.C if action[0][0] == C else Action.D

    def reset(self) -> None:
        super().reset()
        self.efe_decomposition_history = []


### Utility Functions
def plot_A_1(A_1):
    plt.figure(figsize=(4, 4))
    sns.heatmap(A_1, annot=True, cmap='viridis', cbar=False)
    # label axes 
    ticks = [i + 0.5 for i in range(num_states)]
    plt.xticks(ticks=ticks, labels=ALL_STATES_LABELS)
    plt.yticks(ticks=ticks, labels=ALL_STATES_LABELS)
    
    
    plt.title('Transition Matrix A_1')
    plt.xlabel('Observation')
    plt.ylabel('State')
    plt.show()


def plot_B_1(B_1):
    # B_1 Shape (next_state, current_state, action)
    assert B_1.shape == (num_states, num_states, num_actions), f"B_1 shape is {B_1.shape}, expected (num_states, num_states, num_actions)"
    plt.figure(figsize=(8, 4))
    for action in range(num_actions):
        plt.subplot(1, num_actions, action + 1)
        sns.heatmap(B_1[:, :, action], annot=True, cmap='viridis', cbar=False)
        # offset labels on ticks by 0.5
        ticks = [i + 0.5 for i in range(num_states)]
        plt.xticks(ticks=ticks, labels=ALL_STATES_LABELS)
        plt.yticks(ticks=ticks, labels=ALL_STATES_LABELS)
        plt.title(f'B_1 for action {ALL_ACTIONS_LABELS[action]}')
        plt.xlabel('Next State')
        plt.ylabel('Current State')
        plt.tight_layout()
  
    plt.show()


def plot_C_1(C_1, title="Preferences"):
    plt.grid(zorder=0)
    plt.bar(range(C_1.shape[0]), C_1, color='r', zorder=3)
    plt.xticks(range(C_1.shape[0]), ALL_OBS_LABELS)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    import axelrod as axl
    from axelrod import Match
    from axelrod.strategies.titfortat import TitForTat

    # Test standard agent
    agent = ActiveInferenceAgent(
        seed=0, lr_B=1.5, update_interval=10, alpha=0.6, 
        bias=0.5, cooperative_preference=False, pB_scale=1, policy_len=5, action_selection="deterministic"
    )

    gtft = axl.TitForTat()
    match1 = Match((agent, gtft), turns=1000, noise=0.05)
    match1.play()
    print("Standard agent:", match1.final_score())
    
    # Test deterministic agent
    agent2 = ActiveInferenceAgent(
        seed=0, lr_B=1.5, update_interval=10, alpha=0.6, 
        bias=0.5, cooperative_preference=False, pB_scale=1, policy_len=5,
        action_selection="deterministic"
    )
    match2 = Match((agent2, axl.TitForTat()), turns=1000, noise=0.05)
    match2.play()
    print("Deterministic agent:", match2.final_score())
    
    # Test with decay and cooperative preference
    agent3 = ActiveInferenceAgent(
        seed=0, lr_B=1.5, update_interval=10, alpha=0.6, 
        bias=0.5, cooperative_preference=True, pB_scale=1, policy_len=10,
        pB_decay_rate=0.9
    )
    match3 = Match((agent3, axl.Grudger()), turns=1000, noise=0.05)
    match3.play()
    print("Agent with decay and cooperative preference:", match3.final_score())
    
    # Test with noisy observation model
    agent4 = ActiveInferenceAgent(
        seed=0, lr_B=1.5, update_interval=10, alpha=0.6, 
        bias=0.5, cooperative_preference=False, pB_scale=1, policy_len=5,
        action_selection="deterministic", use_noisy_observation_model=True
    )
    match4 = Match((agent4, axl.TitForTat()), turns=1000, noise=0.05)
    match4.play()
    print("Agent with noisy observation model:", match4.final_score())

