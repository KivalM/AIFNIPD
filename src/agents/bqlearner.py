from typing import Union

import jax
import jax.numpy as jnp
from axelrod.action import Action
from axelrod.player import Player
from jax import random as jr

Score = Union[int, float]

C, D = Action.C, Action.D

# States from five_state.py
START = 0
CC = 1
CD = 2
DC = 3
DD = 4

# Actions
COOPERATE = 0
DEFECT = 1


class JaxBayesianQLearner(Player):
    """
    A base class for Jax-based Bayesian Q-learning agents with Thompson sampling.
    """

    name = "Jax Bayesian QLearner"
    classifier = {
        "memory_depth": 1,
        "stochastic": True,
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    discount_rate = 0.9
    initial_variance = 1.0
    reward_variance = 1.0

    def __init__(self, discount_rate: float = 0.9, initial_variance: float = 1.0, reward_variance: float = 1.0) -> None:
        super().__init__()
        self.classifier["stochastic"] = True

        self.mus = jnp.zeros((5, 2))  # Mean of Q-values
        self.sigmas_sq = (
            jnp.ones((5, 2)) * self.initial_variance
        )  # Variance of Q-values

        self.prev_state = START
        self.prev_action = None
        self.set_seed(0)

    def set_seed(self, seed: int):
        self.seed = seed
        self.rng_key = jr.PRNGKey(self.seed)

    def receive_match_attributes(self):
        (R, P, S, T) = self.match_attributes["game"].RPST()
        self.payoff_matrix = {C: {C: R, D: S}, D: {C: T, D: P}}

    @staticmethod
    @jax.jit
    def _update_q_distribution(
        mus, sigmas_sq, prev_state, state, action, reward, discount_rate, reward_variance
    ):
        # Calculate target value y
        Vs_next = jnp.max(mus[state])
        y = reward + discount_rate * Vs_next

        # Update distribution for Q(prev_state, action)
        mu_old = mus[prev_state, action]
        sigma_sq_old = sigmas_sq[prev_state, action]

        # Bayesian update (Kalman filter style)
        new_mu = (reward_variance * mu_old + sigma_sq_old * y) / (
            reward_variance + sigma_sq_old
        )
        new_sigma_sq = (reward_variance * sigma_sq_old) / (
            reward_variance + sigma_sq_old
        )

        mus = mus.at[prev_state, action].set(new_mu)
        sigmas_sq = sigmas_sq.at[prev_state, action].set(new_sigma_sq)

        return mus, sigmas_sq

    def update_q_distribution(self, prev_state, state, action, reward):
        return self._update_q_distribution(
            self.mus,
            self.sigmas_sq,
            prev_state,
            state,
            action,
            reward,
            self.discount_rate,
            self.reward_variance,
        )

    @staticmethod
    @jax.jit
    def _select_action(rng_key, mus, sigmas_sq, state):
        # Thompson sampling
        q_samples = mus[state] + jnp.sqrt(sigmas_sq[state]) * jr.normal(
            rng_key, shape=(2,)
        )
        return jnp.argmax(q_samples)

    def select_action(self, rng_key, state: int) -> int:
        return int(self._select_action(rng_key, self.mus, self.sigmas_sq, state))

    def find_state(self, opponent: Player) -> int:
        if len(self.history) == 0:
            return START
        my_last_action = self.history[-1]
        opponent_last_action = opponent.history[-1]
        state_idx = 1 + (my_last_action.value * 2) + opponent_last_action.value
        return state_idx

    def find_reward(self, opponent: Player):
        if len(opponent.history) == 0:
            return 0

        my_action = Action.C if self.prev_action == COOPERATE else Action.D
        opp_action = opponent.history[-1]
        return self.payoff_matrix[my_action][opp_action]

    def strategy(self, opponent: Player) -> Action:
        self.rng_key, subkey = jr.split(self.rng_key)
        state = self.find_state(opponent)

        if len(self.history) > 0:
            reward = self.find_reward(opponent)
            self.mus, self.sigmas_sq = self.update_q_distribution(
                self.prev_state, state, self.prev_action, reward
            )

        action_idx = self.select_action(subkey, state)
        self.prev_state = state
        self.prev_action = action_idx
        return Action.C if action_idx == COOPERATE else Action.D

    def reset(self) -> None:
        super().reset()
        self.mus = jnp.zeros((5, 2))
        self.sigmas_sq = jnp.ones((5, 2)) * self.initial_variance
        self.prev_state = START
        self.prev_action = None


class RiskyBQLearner(JaxBayesianQLearner):
    name = "Risky BQLearner"
    discount_rate = 0.9
    reward_variance = 0.1  # Analogous to high learning rate


class ArrogantBQLearner(JaxBayesianQLearner):
    name = "Arrogant BQLearner"
    discount_rate = 0.1
    reward_variance = 0.1


class HesitantBQLearner(JaxBayesianQLearner):
    name = "Hesitant BQLearner"
    discount_rate = 0.9
    reward_variance = 1.0  # Analogous to low learning rate


class CautiousBQLearner(JaxBayesianQLearner):
    name = "Cautious BQLearner"
    discount_rate = 0.1
    reward_variance = 1.0


if __name__ == "__main__":
    import axelrod as axl

    agent = HesitantBQLearner()
    opponent = axl.RiskyQLearner()
    match = axl.Match((agent, opponent), turns=1000, noise=0.05)
    match.play()
    print(agent.mus)
    print(agent.sigmas_sq)
    print(match.final_score_per_turn())
