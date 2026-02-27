import jax.numpy as jnp
from jax import jit
from jax import random as jr
from collections import deque
from typing import Optional, Tuple

from axelrod.action import Action

from .aif import (
    make_agent,
    action_pair_to_obs,
    update_B,
    START,
    C,
    D,
    n_batches,
    num_states,
    num_actions,
)
from ...wrapper import JointWrapper


@jit
def step_epsilon_greedy(rng_key, agent, obs_idx, empirical_prior, epsilon):
    """Infer states/policies then select action via epsilon-greedy."""
    qs = agent.infer_states(obs_idx, empirical_prior)
    qpi, _ = agent.infer_policies(qs)
    explore_key, action_key, sample_key = jr.split(rng_key, 3)

    greedy_action = agent.sample_action(qpi, rng_key=jnp.expand_dims(sample_key, 0))
    random_action = jr.randint(action_key, shape=greedy_action.shape, minval=0, maxval=num_actions)

    explore = jr.uniform(explore_key, shape=()) < epsilon
    action = jnp.where(explore, random_action, greedy_action)

    empirical_prior, qs = agent.update_empirical_prior(action, qs)
    return obs_idx, empirical_prior, qs, action


class EpsilonGreedyAIFAgent(JointWrapper):
    """Active inference agent using utility-only EFE with epsilon-greedy exploration."""

    name = "EpsilonGreedyAIFAgent"

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
        noise: float = 0.0,
        use_noisy_observation_model: bool = False,
        pB_decay_rate: float = 1.0,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ) -> None:
        super().__init__()
        self.policy_len = policy_len
        self.update_interval = update_interval
        self.seed = seed
        self.lr_B = lr_B
        self.alpha = alpha
        self.gamma = gamma
        self.bias = bias
        self.cooperative_preference = cooperative_preference
        self.pB_scale = pB_scale
        self.noise = noise
        self.use_noisy_observation_model = use_noisy_observation_model
        self.pB_decay_rate = pB_decay_rate
        self.preference_params = (3.0, 1.0, 0.0, 5.0)

        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.total_steps = 0

        self.qs = deque(maxlen=self.update_interval + 1)
        self.obs = deque(maxlen=self.update_interval + 1)
        self.actions = deque(maxlen=self.update_interval + 1)
        self.step_count = 0

        self.agent = self._make_agent()
        self.empirical_prior = self.agent.D
        self.set_seed(seed)

    def _make_agent(self):
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
            action_selection="deterministic",
            use_utility=True,
            use_states_info_gain=False,
            use_param_info_gain=False,
            preference_params=self.preference_params,
        )

    def receive_match_attributes(self) -> None:
        super().receive_match_attributes()
        if "game" in self.match_attributes:
            self.preference_params = self.match_attributes["game"].RPST()
        if hasattr(self, "agent") and self.agent is not None:
            self.agent = self._make_agent()

    def set_seed(self, seed: int):
        self.seed = seed
        self.rng_key = jr.PRNGKey(self.seed)
        self.rng_key = jr.split(self.rng_key)[1]

    def step(self, state: Tuple[Optional[Action], Optional[Action]]) -> Action:
        if state[0] is None and state[1] is None:
            obs_idx = [jnp.broadcast_to(jnp.array([START]), (1, 1))]
        else:
            obs_idx = [action_pair_to_obs(state[0].value, state[1].value)]

        self.rng_key = jr.split(self.rng_key)[1]
        obs_idx, empirical_prior, qs, action = step_epsilon_greedy(
            self.rng_key, self.agent, obs_idx, self.empirical_prior, self.epsilon
        )
        self.empirical_prior = empirical_prior

        self.qs.append(qs)
        self.obs.append(obs_idx)
        self.actions.append(action)
        self.step_count += 1
        self.total_steps += 1

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon_start * (self.epsilon_decay ** self.total_steps),
        )

        if (
            self.step_count >= self.update_interval
            and len(self.actions) >= self.update_interval
            and len(self.qs) >= self.update_interval + 1
        ):
            qs_arr = jnp.array([q for q in self.qs])
            obs_arr = jnp.array([o for o in self.obs])
            act_arr = jnp.array([a for a in self.actions])

            beliefs = [qs_arr.reshape(n_batches, len(self.qs), num_states)]
            outcomes = obs_arr.reshape(n_batches, len(self.obs))
            actions = act_arr[:-1].reshape(n_batches, len(self.actions) - 1, 1)

            self.agent = update_B(
                self.agent, beliefs, outcomes, actions,
                lr_B=self.lr_B, pB_decay_rate=self.pB_decay_rate,
            )
            self.step_count = 0

        return Action.C if action[0][0] == C else Action.D

    def reset(self) -> None:
        self.qs = deque(maxlen=self.update_interval + 1)
        self.obs = deque(maxlen=self.update_interval + 1)
        self.actions = deque(maxlen=self.update_interval + 1)
        self.step_count = 0
        self.total_steps = 0
        self.epsilon = self.epsilon_start
        self.empirical_prior = self.agent.D
        super().reset()


if __name__ == "__main__":
    import axelrod as axl
    from axelrod import Match

    agent = EpsilonGreedyAIFAgent(
        seed=0, lr_B=1.5, update_interval=10, alpha=0.6,
        bias=0.5, cooperative_preference=False, pB_scale=1, policy_len=5,
        epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995,
    )
    match1 = Match((agent, axl.TitForTat()), turns=1000, noise=0.05)
    match1.play()
    print("Epsilon-greedy vs TFT:", match1.final_score())

    agent2 = EpsilonGreedyAIFAgent(
        seed=0, lr_B=1.5, update_interval=10, alpha=0.6,
        bias=0.5, cooperative_preference=False, pB_scale=1, policy_len=5,
        epsilon_start=0.5, epsilon_min=0.01, epsilon_decay=0.99,
    )
    match2 = Match((agent2, axl.Grudger()), turns=1000, noise=0.05)
    match2.play()
    print("Epsilon-greedy (low eps) vs Grudger:", match2.final_score())

    agent3 = EpsilonGreedyAIFAgent(
        seed=0, lr_B=1.5, update_interval=10, alpha=0.6,
        bias=0.5, cooperative_preference=True, pB_scale=1, policy_len=10,
        pB_decay_rate=0.9,
        epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.99,
    )
    match3 = Match((agent3, axl.WinStayLoseShift()), turns=1000, noise=0.05)
    match3.play()
    print("Epsilon-greedy (coop) vs WSLS:", match3.final_score())
