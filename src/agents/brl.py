from collections import OrderedDict
from typing import Dict, Union

import jax
import jax.numpy as jnp
from axelrod.action import Action, actions_to_str
from axelrod.player import Player
from jax import jit

C, D = Action.C, Action.D
Score = Union[int, float]


@jit
def _perform_bayesian_q_learning(
    mu_prev,
    sigma_sq_prev,
    mu_next,
    sigma_sq_next,
    reward,
    discount_rate,
    reward_variance,
):
    """
    Performs the Bayesian Q-learning update using JAX.
    """
    target_mean = reward + discount_rate * mu_next
    target_variance = reward_variance + (discount_rate**2) * sigma_sq_next

    precision_prev = 1.0 / sigma_sq_prev
    precision_target = 1.0 / target_variance

    precision_posterior = precision_prev + precision_target
    sigma_sq_posterior = 1.0 / precision_posterior

    mu_posterior = sigma_sq_posterior * (
        precision_prev * mu_prev + precision_target * target_mean
    )

    return mu_posterior, sigma_sq_posterior


class BayesianQLearner(Player):
    """A player who learns the best strategies through a Bayesian Q-learning
    algorithm."""

    name = "Bayesian QLearner"
    classifier = {
        "memory_depth": float("inf"),
        "stochastic": True,
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    # Parameters
    discount_rate = 0.95
    memory_length = 10
    # Prior distribution parameters for Q-values
    mu_0 = 0.0  # Initial mean
    sigma_0 = 1.0  # Initial standard deviation
    reward_variance = 1.0  # Assumed variance of rewards

    def __init__(self, seed=None) -> None:
        """Initialises the player."""

        super().__init__()

        self.classifier["stochastic"] = True

        self.prev_action = None  # type: Action
        # Qs will store tuples of (mean, variance) for each state-action pair
        self.Qs = OrderedDict(
            {
                "": OrderedDict(
                    zip(
                        [C, D],
                        [
                            (self.mu_0, self.sigma_0**2),
                            (self.mu_0, self.sigma_0**2),
                        ],
                    )
                )
            }
        )
        self.prev_state = ""
        # For analysis
        self.history_of_q_values = []
        self.key = jax.random.PRNGKey(seed if seed is not None else 0)

    def receive_match_attributes(self):
        (R, P, S, T) = self.match_attributes["game"].RPST()
        self.payoff_matrix = {C: {C: R, D: S}, D: {C: T, D: P}}

    def strategy(self, opponent: Player) -> Action:
        """Runs a Bayesian Q-learning algorithm."""
        if len(self.history) == 0:
            self.prev_action = self._random.random_choice()

        state = self.find_state(opponent)
        reward = self.find_reward(opponent)

        if state not in self.Qs:
            self.Qs[state] = OrderedDict(
                zip(
                    [C, D],
                    [(self.mu_0, self.sigma_0**2), (self.mu_0, self.sigma_0**2)],
                )
            )

        if len(self.history) > 0:
            self.perform_bayesian_q_learning(
                self.prev_state, state, self.prev_action, reward
            )

        action = self.select_action(state)
        self.prev_state = state
        self.prev_action = action

        # Store Q-values for analysis
        self.history_of_q_values.append(
            {
                s: {a: v for a, v in av.items()}
                for s, av in self.Qs.items()
            }
        )

        return action

    def select_action(self, state: str) -> Action:
        """
        Selects the action based on Thompson sampling.
        """
        samples = {}
        for action in self.Qs[state]:
            mu, sigma_sq = self.Qs[state][action]
            self.key, subkey = jax.random.split(self.key)
            samples[action] = mu + jnp.sqrt(sigma_sq) * jax.random.normal(
                subkey
            )

        return max(samples, key=samples.get)

    def find_state(self, opponent: Player) -> str:
        """
        Finds the state (the opponents last n moves +
        its previous proportion of playing C) as a hashable state
        """
        if len(opponent.history) == 0:
            prob = "0.0"
        else:
            prob = "{:.1f}".format(opponent.cooperations / len(opponent.history))
        action_str = actions_to_str(opponent.history[-self.memory_length :])
        return action_str + prob

    def perform_bayesian_q_learning(
        self, prev_state: str, state: str, action: Action, reward: float
    ):
        """
        Performs the Bayesian Q-learning update
        """
        best_next_action = max(
            self.Qs[state], key=lambda act: self.Qs[state][act][0]
        )
        mu_next, sigma_sq_next = self.Qs[state][best_next_action]
        mu_prev, sigma_sq_prev = self.Qs[prev_state][action]

        mu_posterior, sigma_sq_posterior = _perform_bayesian_q_learning(
            mu_prev,
            sigma_sq_prev,
            mu_next,
            sigma_sq_next,
            reward,
            self.discount_rate,
            self.reward_variance,
        )

        self.Qs[prev_state][action] = (
            mu_posterior.item(),
            sigma_sq_posterior.item(),
        )

    def find_reward(self, opponent: Player) -> Score:
        """
        Finds the reward gained on the last iteration
        """
        if len(self.history) == 0:
            return 0

        my_action = self.history[-1]
        opponent_action = opponent.history[-1]
        return self.payoff_matrix[my_action][opponent_action]

    def get_q_value_history(self, state, action):
        """
        Utility function to retrieve the history of mean and variance for a
        given state-action pair.
        """
        history = []
        for q_snapshot in self.history_of_q_values:
            if state in q_snapshot and action in q_snapshot[state]:
                history.append(q_snapshot[state][action])
        return history

    @staticmethod
    def plot_q_value_history(q_history):
        """
        Utility function to plot the history of Q-value distribution.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is not installed. Cannot plot Q-value history.")
            return

        means = [item[0] for item in q_history]
        variances = [item[1] for item in q_history]
        stds = jnp.sqrt(jnp.array(variances))
        iterations = range(len(means))

        plt.figure(figsize=(10, 6))
        plt.plot(iterations, means, label="Mean Q-value")
        plt.fill_between(
            iterations,
            jnp.array(means) - 2 * stds,
            jnp.array(means) + 2 * stds,
            alpha=0.2,
            label="95% confidence interval",
        )
        plt.xlabel("Iteration")
        plt.ylabel("Q-value")
        plt.title("Evolution of Q-value distribution")
        plt.legend()
        plt.grid(True)
        plt.show()

    def full_reset(self):
        """Resets all attributes to their initial states."""
        self.__init__(seed=self.key[0].item())
        self.history = []
        self.match_attributes = {}


class AggressiveBayesianQLearner(BayesianQLearner):
    """
    An aggressive variant of the Bayesian Q-learner.
    - Low initial mean Q-value for cooperation, high for defection.
    - High discount rate, prioritizing future rewards.
    - Short memory length, focusing on recent opponent actions.
    """

    name = "Aggressive Bayesian QLearner"

    def __init__(self, seed=None) -> None:
        super().__init__(seed)
        self.discount_rate = 0.99
        self.memory_length = 2
        self.Qs = OrderedDict(
            {
                "": OrderedDict(
                    zip(
                        [C, D],
                        [
                            (-1.0, self.sigma_0**2),
                            (1.0, self.sigma_0**2),
                        ],
                    )
                )
            }
        )


class CooperativeBayesianQLearner(BayesianQLearner):
    """
    A cooperative variant of the Bayesian Q-learner.
    - High initial mean Q-value for cooperation, low for defection.
    - High discount rate, valuing long-term cooperation.
    - Long memory length, considering a longer history of interactions.
    """

    name = "Cooperative Bayesian QLearner"

    def __init__(self, seed=None) -> None:
        super().__init__(seed)
        self.discount_rate = 0.99
        self.memory_length = 15
        self.Qs = OrderedDict(
            {
                "": OrderedDict(
                    zip(
                        [C, D],
                        [
                            (1.0, self.sigma_0**2),
                            (-1.0, self.sigma_0**2),
                        ],
                    )
                )
            }
        )


class CautiousBayesianQLearner(BayesianQLearner):
    """
    A cautious variant of the Bayesian Q-learner.
    - Low initial uncertainty (sigma_0), making it less exploratory.
    - Low reward variance, assuming a more predictable environment.
    - Medium memory length, balancing recent and past history.
    """

    name = "Cautious Bayesian QLearner"

    def __init__(self, seed=None) -> None:
        super().__init__(seed)
        self.sigma_0 = 0.25
        self.reward_variance = 0.25
        self.memory_length = 8


class ErraticBayesianQLearner(BayesianQLearner):
    """
    An erratic variant of the Bayesian Q-learner.
    - High initial uncertainty (sigma_0), promoting exploration.
    - High reward variance, assuming a noisy/unpredictable environment.
    - Short memory length, leading to more reactive behavior.
    """

    name = "Erratic Bayesian QLearner"

    def __init__(self, seed=None) -> None:
        super().__init__(seed)
        self.sigma_0 = 2.0
        self.reward_variance = 2.0
        self.memory_length = 3
