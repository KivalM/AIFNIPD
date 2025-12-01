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
@partial(jax.jit, static_argnames=("learning_rate", "discount_rate", "action_selection_parameter", "min_epsilon", "decay_rate", "R", "P", "S", "T"))
def _strategy_step(
    # State variables
    Qs,
    Vs,
    prev_state,
    prev_action,
    rng_key,
    epsilon,
    # Information from the current turn
    current_state,
    is_first_turn,
    opponent_last_action,
    # Hyperparameters (static)
    learning_rate,
    discount_rate,
    action_selection_parameter,
    min_epsilon,
    decay_rate,
    R, P, S, T,
):
    """
    Performs all logic for a single turn within a JIT-compiled function.
    """
    # Create the payoff matrix inside the function
    payoff_matrix = jnp.array([[R, S], [T, P]], dtype=DTYPE)
    
    # 1. Determine the reward based on the last turn
    my_last_action = jnp.array(prev_action, dtype=jnp.int32)
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

    # 3. Select the next action using epsilon-greedy (with decaying epsilon)
    rng_key, subkey1, subkey2 = jr.split(rng_key, 3)
    p = 1.0 - epsilon
    exploit = jr.uniform(subkey1) < p

    action_idx = lax.cond(
        exploit,
        lambda: jnp.argmax(Qs[state_for_action_selection]),
        lambda: jr.choice(subkey2, jnp.array([COOPERATE, DEFECT])),
    )

    # 4. Calculate next epsilon
    next_epsilon = jnp.maximum(min_epsilon, epsilon * decay_rate)

    # 5. Return all updated state variables
    return Qs, Vs, state_for_action_selection, action_idx, rng_key, next_epsilon, action_idx


@partial(jax.jit, static_argnames=("learning_rate", "discount_rate", "action_selection_parameter", "min_epsilon", "decay_rate", "R", "P", "S", "T"))
def _cooperative_strategy_step(
    # State variables
    Qs,
    Vs,
    prev_state,
    prev_action,
    rng_key,
    epsilon,
    # Information from the current turn
    current_state,
    is_first_turn,
    opponent_last_action,
    # Hyperparameters (static)
    learning_rate,
    discount_rate,
    action_selection_parameter,
    min_epsilon,
    decay_rate,
    R, P, S, T,
):
    """
    Performs all logic for a single turn within a JIT-compiled function.
    Modified version for cooperative Q-learning where DC reward is 0.
    """
    # Create the payoff matrix with DC reward set to 0
    payoff_matrix = jnp.array([[R, S], [0, P]], dtype=DTYPE)
    
    # 1. Determine the reward based on the last turn
    my_last_action = jnp.array(prev_action, dtype=jnp.int32)
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

    # 3. Select the next action using epsilon-greedy (with decaying epsilon)
    rng_key, subkey1, subkey2 = jr.split(rng_key, 3)
    p = 1.0 - epsilon
    exploit = jr.uniform(subkey1) < p

    action_idx = lax.cond(
        exploit,
        lambda: jnp.argmax(Qs[state_for_action_selection]),
        lambda: jr.choice(subkey2, jnp.array([COOPERATE, DEFECT])),
    )

    # 4. Calculate next epsilon
    next_epsilon = jnp.maximum(min_epsilon, epsilon * decay_rate)

    # 5. Return all updated state variables
    return Qs, Vs, state_for_action_selection, action_idx, rng_key, next_epsilon, action_idx


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
    min_epsilon = 0.01
    decay_rate = 0.999

    def __init__(self, learning_rate: float = 0.5, discount_rate: float = 0.9, action_selection_parameter: float = 0.1, memory_length: int = 1, decay_rate: float = 0.999) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.action_selection_parameter = action_selection_parameter
        self.memory_length = memory_length
        self.decay_rate = decay_rate
        self.classifier["stochastic"] = True
        self.set_seed(0)
        self.init_state()

    def set_seed(self, seed: int):
        self.seed = seed
        self.rng_key = jr.PRNGKey(self.seed)

    def init_state(self):
        """Initializes or resets the agent's state."""
        num_states = 4**self.memory_length + 1
        self.Qs = jnp.zeros((num_states, 2), dtype=DTYPE)
        self.Vs = jnp.zeros(num_states, dtype=DTYPE)
        self.prev_state = START
        # Initialize prev_action to Cooperate for the first update step
        self.prev_action = COOPERATE
        # Initialize epsilon
        self.epsilon = self.action_selection_parameter

    def receive_match_attributes(self):
        (self.R, self.P, self.S, self.T) = self.match_attributes["game"].RPST()

    def _calculate_state(self, opponent: Player) -> int:
        """Calculates the state index based on interaction history."""
        if len(self.history) < self.memory_length:
            return START
        
        state_idx = 0
        # Iterate backwards from the most recent turn
        for i in range(self.memory_length):
            # Access from the end: -1 is last turn, -2 is turn before that
            turn_idx = -1 - i
            
            # Use .value to get 0 (C) or 1 (D)
            my_action = int(self.history[turn_idx].value)
            opp_action = int(opponent.history[turn_idx].value)
            
            # Encode: 0=CC, 1=CD, 2=DC, 3=DD
            # COOPERATE=0, DEFECT=1
            code = (my_action * 2) + opp_action
            state_idx += code * (4**i)
            
        return 1 + state_idx

    def strategy(self, opponent: Player) -> Action:
        is_first_turn = len(self.history) == 0
        
        # Get opponent's last action, default to Cooperate if first turn
        opponent_last_action = opponent.history[-1].value if not is_first_turn else COOPERATE

        current_state = self._calculate_state(opponent)

        # Call the single, fast JIT'd function
        new_Qs, new_Vs, state_for_selection, action_idx, new_rng_key, new_epsilon, new_action = _strategy_step(
            self.Qs,
            self.Vs,
            self.prev_state,
            self.prev_action,
            self.rng_key,
            self.epsilon,
            current_state,
            is_first_turn,
            opponent_last_action,
            self.learning_rate,
            self.discount_rate,
            self.action_selection_parameter,
            self.min_epsilon,
            self.decay_rate,
            self.R, self.P, self.S, self.T,
        )

        # Update the agent's state with the results from the pure function
        self.Qs = new_Qs
        self.Vs = new_Vs
        self.rng_key = new_rng_key
        self.epsilon = new_epsilon
        self.prev_state = int(state_for_selection)
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

class CooperativeQLearner(JaxQLearner):
    """
    A cooperative Q-learning variant where exploitation (DC) gives 0 reward,
    making cooperation (CC) the highest reward outcome.
    """
    name = "Cooperative QLearner"
    
    def strategy(self, opponent: Player) -> Action:
        is_first_turn = len(self.history) == 0
        
        # Get opponent's last action, default to Cooperate if first turn
        opponent_last_action = opponent.history[-1].value if not is_first_turn else COOPERATE

        current_state = self._calculate_state(opponent)

        # Call the single, fast JIT'd function with modified payoff matrix
        new_Qs, new_Vs, state_for_selection, action_idx, new_rng_key, new_epsilon, new_action = _cooperative_strategy_step(
            self.Qs,
            self.Vs,
            self.prev_state,
            self.prev_action,
            self.rng_key,
            self.epsilon,
            current_state,
            is_first_turn,
            opponent_last_action,
            self.learning_rate,
            self.discount_rate,
            self.action_selection_parameter,
            self.min_epsilon,
            self.decay_rate,
            self.R, self.P, self.S, self.T,
        )

        # Update the agent's state with the results from the pure function
        self.Qs = new_Qs
        self.Vs = new_Vs
        self.rng_key = new_rng_key
        self.epsilon = new_epsilon
        self.prev_state = int(state_for_selection)
        self.prev_action = int(new_action)
        
        return C if action_idx == COOPERATE else D

if __name__ == "__main__":
    import axelrod as axl
    import time
    
    # Test the QLearner
    agent = JaxQLearner(0.9, 0.9, 0.1, 1)
    opponent = JaxQLearner(0.9, 0.9, 0.1, 1)

    
    
    match = axl.Match((agent, opponent), turns=1000, noise=0.00) # Increased turns to see the speedup
    match.play()
 
    print(f"Final score per turn: {match.final_score_per_turn()}")
    average_score = sum(match.final_score_per_turn()) / len(match.final_score_per_turn())
    print(f"Average score per turn: {average_score}")
    
        # Test the QLearner
    agent = JaxQLearner(0.9, 0.9, 0.1, 2)
    opponent = JaxQLearner(0.9, 0.9, 0.1, 2)

    
    
    match = axl.Match((agent, opponent), turns=1000, noise=0.00) # Increased turns to see the speedup
    match.play()
 
    print(f"Final score per turn: {match.final_score_per_turn()}")
    average_score = sum(match.final_score_per_turn()) / len(match.final_score_per_turn())
    print(f"Average score per turn: {average_score}")

            # Test the QLearner
    agent = JaxQLearner(0.9, 0.9, 0.1, 3)
    opponent = JaxQLearner(0.9, 0.9, 0.1, 3)

    
    
    match = axl.Match((agent, opponent), turns=1000, noise=0.00) # Increased turns to see the speedup
    match.play()
 
    print(f"Final score per turn: {match.final_score_per_turn()}")
    average_score = sum(match.final_score_per_turn()) / len(match.final_score_per_turn())
    print(f"Average score per turn: {average_score}")

                # Test the QLearner
    agent = JaxQLearner(0.9, 0.9, 0.1, 4)
    opponent = JaxQLearner(0.9, 0.9, 0.1, 4)

    
    
    match = axl.Match((agent, opponent), turns=1000, noise=0.00) # Increased turns to see the speedup
    match.play()
 
    print(f"Final score per turn: {match.final_score_per_turn()}")
    average_score = sum(match.final_score_per_turn()) / len(match.final_score_per_turn())
    print(f"Average score per turn: {average_score}")

                    # Test the QLearner
    agent = JaxQLearner(0.9, 0.9, 0.1, 5)
    opponent = JaxQLearner(0.9, 0.9, 0.1, 5)

    
    
    match = axl.Match((agent, opponent), turns=1000, noise=0.00) # Increased turns to see the speedup
    match.play()
 
    print(f"Final score per turn: {match.final_score_per_turn()}")
    average_score = sum(match.final_score_per_turn()) / len(match.final_score_per_turn())
    print(f"Average score per turn: {average_score}")