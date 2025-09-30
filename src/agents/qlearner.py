from typing import Dict, Union

import jax
import jax.numpy as jnp
from axelrod.action import Action
from axelrod.player import Player
from jax import random as jr
from jax import lax

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


class JaxQLearner(Player):
    """
    A base class for Jax-based Q-learning agents.
    """

    learning_rate = 0.5
    discount_rate = 0.9
    action_selection_parameter = 0.1
    name = "QLearner"
    def __init__(self, learning_rate: float = 0.5, discount_rate: float = 0.9, action_selection_parameter: float = 0.1) -> None:
        super().__init__()
        self.classifier["stochastic"] = True

        self.Qs = jnp.zeros((5, 2))
        self.Vs = jnp.zeros(5)
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
    def _perform_q_learning(
        Qs, Vs, prev_state, state, action, reward, learning_rate, discount_rate
    ):
        q_value = (1.0 - learning_rate) * Qs[prev_state, action] + learning_rate * (
            reward + discount_rate * Vs[state]
        )
        Qs = Qs.at[prev_state, action].set(q_value)
        Vs = Vs.at[prev_state].set(jnp.max(Qs[prev_state]))
        return Qs, Vs

    def perform_q_learning(self, prev_state, state, action, reward):
        return self._perform_q_learning(
            self.Qs,
            self.Vs,
            prev_state,
            state,
            action,
            reward,
            self.learning_rate,
            self.discount_rate,
        )

    def select_action(self, rng_key, state: int) -> int:
        subkey1, subkey2 = jr.split(rng_key)
        rnd_num = jr.uniform(subkey1)
        p = 1.0 - self.action_selection_parameter

        exploit = rnd_num < p
        
        action = lax.cond(
            exploit,
            lambda: jnp.argmax(self.Qs[state]),
            lambda: jr.choice(subkey2, jnp.array([COOPERATE, DEFECT])),
        )
        return int(action)

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
            self.Qs, self.Vs = self.perform_q_learning(
                self.prev_state, state, self.prev_action, reward
            )

        action_idx = self.select_action(subkey, state)
        self.prev_state = state
        self.prev_action = action_idx
        return Action.C if action_idx == COOPERATE else Action.D

    def reset(self) -> None:
        super().reset()
        self.Qs = jnp.zeros((5, 2))
        self.Vs = jnp.zeros(5)
        self.prev_state = START
        self.prev_action = None


class RiskyQLearner(JaxQLearner):
    name = "Risky QLearner"
    classifier = {
        "memory_depth": 1,
        "stochastic": True,
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }
    learning_rate = 0.9
    discount_rate = 0.9


class ArrogantQLearner(RiskyQLearner):
    name = "Arrogant QLearner"
    learning_rate = 0.9
    discount_rate = 0.1


class HesitantQLearner(RiskyQLearner):
    name = "Hesitant QLearner"
    learning_rate = 0.5
    discount_rate = 0.9


class CautiousQLearner(RiskyQLearner):
    name = "Cautious QLearner"
    learning_rate = 0.1
    discount_rate = 0.1

if __name__ == "__main__":
    import axelrod as axl
    agent = HesitantQLearner()
    opponent = axl.Defector()
    match = axl.Match((agent, opponent), turns=1000, noise=0.05)
    match.play()
    print(match.final_score_per_turn())