from typing import Union
from functools import partial

import jax
import jax.numpy as jnp
from axelrod.action import Action
from axelrod.player import Player
from jax import random as jr
from jax import lax

# Use a more memory-efficient dtype
# JAX often defaults to 64-bit, but 32-bit is usually sufficient and faster.
DTYPE = jnp.float32

# Define actions and states as constants
C, D = Action.C, Action.D
COOPERATE, DEFECT = 0, 1
START, CC, CD, DC, DD = 0, 1, 2, 3, 4


# This is our single, pure JIT-compiled function for one turn's logic.
# We use `static_argnames` to tell JAX which arguments are compile-time constants.
# This results in more efficient code.
@partial(jax.jit, static_argnames=("learning_rate", "discount_rate", "action_selection_parameter", "R", "P", "S", "T"))
def _strategy_step(
    # State variables
    Qs,
    Vs,
    prev_state,
    prev_action,
    rng_key,
    # Information from the current turn
    is_first_turn,
    opponent_last_action,
    # Hyperparameters (static)
    learning_rate,
    discount_rate,
    action_selection_parameter,
    R, P, S, T,
):
    """
    Performs all logic for a single turn within a JIT-compiled function.
    """
    # Create the payoff matrix inside the function
    payoff_matrix = jnp.array([[R, S], [T, P]], dtype=DTYPE)
    
    # 1. Determine the current state and reward based on the last turn
    my_last_action = jnp.array(prev_action, dtype=jnp.int32)
    current_state = 1 + (my_last_action * 2) + opponent_last_action
    reward = payoff_matrix[my_last_action, opponent_last_action]

    # 2. Perform Q-Learning Update (from the previous turn's state/action)
    q_value = (1.0 - learning_rate) * Qs[prev_state, prev_action] + learning_rate * (
        reward + discount_rate * Vs[current_state]
    )
    Qs = Qs.at[prev_state, prev_action].set(q_value)
    Vs = Vs.at[prev_state].set(jnp.max(Qs[prev_state]))

    # If it's the first turn, we don't learn, we just select an action from START state
    # We use lax.cond to avoid breaking the JIT compilation
    Qs, Vs, state_for_action_selection = lax.cond(
        is_first_turn,
        lambda: (jnp.zeros_like(Qs), jnp.zeros_like(Vs), START), # Reset and use START state
        lambda: (Qs, Vs, current_state), # Use updated tables and current state
    )

    # 3. Select the next action using epsilon-greedy
    rng_key, subkey1, subkey2 = jr.split(rng_key, 3)
    p = 1.0 - action_selection_parameter
    exploit = jr.uniform(subkey1) < p

    action_idx = lax.cond(
        exploit,
        lambda: jnp.argmax(Qs[state_for_action_selection]),
        lambda: jr.choice(subkey2, jnp.array([COOPERATE, DEFECT])),
    )

    # 4. Return all updated state variables
    return Qs, Vs, state_for_action_selection, action_idx, rng_key, action_idx


class JaxQLearner(Player):
    """
    An optimized Jax-based Q-learning agent that uses a single JIT-compiled
    function for its turn-by-turn logic.
    """
    
    # Set default hyperparameters
    learning_rate = 0.5
    discount_rate = 0.9
    action_selection_parameter = 0.1
    name = "Jax QLearner"

    def __init__(self, learning_rate: float = 0.5, discount_rate: float = 0.9, action_selection_parameter: float = 0.1) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.action_selection_parameter = action_selection_parameter
        self.classifier["stochastic"] = True
        self.set_seed(0)
        self.init_state()

    def set_seed(self, seed: int):
        self.seed = seed
        self.rng_key = jr.PRNGKey(self.seed)

    def init_state(self):
        """Initializes or resets the agent's state."""
        self.Qs = jnp.zeros((5, 2), dtype=DTYPE)
        self.Vs = jnp.zeros(5, dtype=DTYPE)
        self.prev_state = START
        # Initialize prev_action to Cooperate for the first update step
        self.prev_action = COOPERATE

    def receive_match_attributes(self):
        (self.R, self.P, self.S, self.T) = self.match_attributes["game"].RPST()

    def strategy(self, opponent: Player) -> Action:
        is_first_turn = len(self.history) == 0
        
        # Get opponent's last action, default to Cooperate if first turn
        opponent_last_action = opponent.history[-1].value if not is_first_turn else COOPERATE

        # Call the single, fast JIT'd function
        new_Qs, new_Vs, current_state, action_idx, new_rng_key, new_action = _strategy_step(
            self.Qs,
            self.Vs,
            self.prev_state,
            self.prev_action,
            self.rng_key,
            is_first_turn,
            opponent_last_action,
            self.learning_rate,
            self.discount_rate,
            self.action_selection_parameter,
            self.R, self.P, self.S, self.T,
        )

        # Update the agent's state with the results from the pure function
        self.Qs = new_Qs
        self.Vs = new_Vs
        self.rng_key = new_rng_key
        self.prev_state = int(current_state)
        self.prev_action = int(new_action)
        
        return C if action_idx == COOPERATE else D

    def reset(self) -> None:
        super().reset()
        self.init_state()

class RiskyQLearner(JaxQLearner):
    name = "Risky QLearner"
    learning_rate = 0.9
    discount_rate = 0.9

class ArrogantQLearner(RiskyQLearner):
    name = "Arrogant QLearner"
    discount_rate = 0.1

class HesitantQLearner(RiskyQLearner):
    name = "Hesitant QLearner"
    learning_rate = 0.5

class CautiousQLearner(RiskyQLearner):
    name = "Cautious QLearner"
    learning_rate = 0.1
    discount_rate = 0.1

if __name__ == "__main__":
    import axelrod as axl
    import time
    
    agent = HesitantQLearner()
    opponent = axl.Defector()
    
    start_time = time.time()
    match = axl.Match((agent, opponent), turns=10000, noise=0.05) # Increased turns to see the speedup
    match.play()
    end_time = time.time()
    
    print(f"Match finished in {end_time - start_time:.4f} seconds.")
    print(f"Final score per turn: {match.final_score_per_turn()}")