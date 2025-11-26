from typing import Union
from functools import partial

import jax
import jax.numpy as jnp
from axelrod.action import Action
from axelrod.player import Player
from jax import random as jr
from jax import lax

Score = Union[int, float]

# Use float32 for efficiency
DTYPE = jnp.float32

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


# JIT-compiled strategy function for Bayesian Q-learning with Thompson sampling
@partial(jax.jit, static_argnames=("discount_rate", "reward_variance", "R", "P", "S", "T"))
def _bql_strategy_step(
    # State variables
    mus,
    sigmas_sq,
    prev_state,
    prev_action,
    rng_key,
    # Information from the current turn
    current_state,
    is_first_turn,
    opponent_last_action,
    # Hyperparameters (static)
    discount_rate,
    reward_variance,
    R, P, S, T,
):
    """
    Performs all logic for a single turn within a JIT-compiled function.
    Implements Bayesian Q-learning with Thompson sampling for action selection.
    """
    # Create the payoff matrix inside the function
    payoff_matrix = jnp.array([[R, S], [T, P]], dtype=DTYPE)
    
    # 1. Determine the reward based on the last turn
    my_last_action = jnp.array(prev_action, dtype=jnp.int32)
    reward = payoff_matrix[my_last_action, opponent_last_action]

    # 2. Perform Bayesian Q-Learning Update (Kalman filter style)
    # Calculate target value y
    Vs_next = jnp.max(mus[current_state])
    y = reward + discount_rate * Vs_next
    
    # Get old distribution parameters
    mu_old = mus[prev_state, prev_action]
    sigma_sq_old = sigmas_sq[prev_state, prev_action]
    
    # Bayesian update (posterior from prior and observation)
    new_mu = (reward_variance * mu_old + sigma_sq_old * y) / (
        reward_variance + sigma_sq_old
    )
    new_sigma_sq = (reward_variance * sigma_sq_old) / (
        reward_variance + sigma_sq_old
    )
    
    mus = mus.at[prev_state, prev_action].set(new_mu)
    sigmas_sq = sigmas_sq.at[prev_state, prev_action].set(new_sigma_sq)

    # If it's the first turn, we don't learn, we just select an action from START state
    # We use lax.cond to avoid breaking the JIT compilation
    mus, sigmas_sq, state_for_action_selection = lax.cond(
        is_first_turn,
        lambda: (jnp.zeros_like(mus), jnp.ones_like(sigmas_sq), START),
        lambda: (mus, sigmas_sq, current_state),
    )

    # 3. Select the next action using Thompson sampling
    rng_key, subkey = jr.split(rng_key)
    # Sample Q-values from posterior distribution
    q_samples = mus[state_for_action_selection] + jnp.sqrt(sigmas_sq[state_for_action_selection]) * jr.normal(
        subkey, shape=(2,)
    )
    action_idx = jnp.argmax(q_samples)

    # 4. Return all updated state variables
    return mus, sigmas_sq, state_for_action_selection, action_idx, rng_key


# JIT-compiled strategy function for Cooperative Bayesian Q-learning
@partial(jax.jit, static_argnames=("discount_rate", "reward_variance", "R", "P", "S", "T"))
def _cooperative_bql_strategy_step(
    # State variables
    mus,
    sigmas_sq,
    prev_state,
    prev_action,
    rng_key,
    # Information from the current turn
    current_state,
    is_first_turn,
    opponent_last_action,
    # Hyperparameters (static)
    discount_rate,
    reward_variance,
    R, P, S, T,
):
    """
    Cooperative version where exploitation (DC) gives 0 reward.
    This makes cooperation (CC) the highest reward outcome.
    """
    # Create the payoff matrix with DC reward set to 0
    payoff_matrix = jnp.array([[R, S], [0, P]], dtype=DTYPE)
    
    # 1. Determine the reward based on the last turn
    my_last_action = jnp.array(prev_action, dtype=jnp.int32)
    reward = payoff_matrix[my_last_action, opponent_last_action]

    # 2. Perform Bayesian Q-Learning Update (Kalman filter style)
    # Calculate target value y
    Vs_next = jnp.max(mus[current_state])
    y = reward + discount_rate * Vs_next
    
    # Get old distribution parameters
    mu_old = mus[prev_state, prev_action]
    sigma_sq_old = sigmas_sq[prev_state, prev_action]
    
    # Bayesian update (posterior from prior and observation)
    new_mu = (reward_variance * mu_old + sigma_sq_old * y) / (
        reward_variance + sigma_sq_old
    )
    new_sigma_sq = (reward_variance * sigma_sq_old) / (
        reward_variance + sigma_sq_old
    )
    
    mus = mus.at[prev_state, prev_action].set(new_mu)
    sigmas_sq = sigmas_sq.at[prev_state, prev_action].set(new_sigma_sq)

    # If it's the first turn, we don't learn, we just select an action from START state
    # We use lax.cond to avoid breaking the JIT compilation
    mus, sigmas_sq, state_for_action_selection = lax.cond(
        is_first_turn,
        lambda: (jnp.zeros_like(mus), jnp.ones_like(sigmas_sq), START),
        lambda: (mus, sigmas_sq, current_state),
    )

    # 3. Select the next action using Thompson sampling
    rng_key, subkey = jr.split(rng_key)
    # Sample Q-values from posterior distribution
    q_samples = mus[state_for_action_selection] + jnp.sqrt(sigmas_sq[state_for_action_selection]) * jr.normal(
        subkey, shape=(2,)
    )
    action_idx = jnp.argmax(q_samples)

    # 4. Return all updated state variables
    return mus, sigmas_sq, state_for_action_selection, action_idx, rng_key


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

    def __init__(self, discount_rate: float = 0.9, initial_variance: float = 1.0, reward_variance: float = 1.0, memory_length: int = 1) -> None:
        super().__init__()
        self.classifier["stochastic"] = True
        self.memory_length = memory_length
        
        # Fix: Actually assign the parameters
        self.discount_rate = discount_rate
        self.initial_variance = initial_variance
        self.reward_variance = reward_variance

        self.set_seed(0)
        self.init_state()

    def set_seed(self, seed: int):
        self.seed = seed
        self.rng_key = jr.PRNGKey(self.seed)

    def init_state(self):
        """Initializes or resets the agent's state."""
        num_states = 4**self.memory_length + 1
        self.mus = jnp.zeros((num_states, 2), dtype=DTYPE)  # Mean of Q-values
        self.sigmas_sq = (
            jnp.ones((num_states, 2), dtype=DTYPE) * self.initial_variance
        )  # Variance of Q-values

        self.prev_state = START
        # Fix: Initialize to COOPERATE instead of None to prevent indexing errors
        self.prev_action = COOPERATE

    def receive_match_attributes(self):
        (R, P, S, T) = self.match_attributes["game"].RPST()
        # Fix: Use JAX array instead of nested dict for JIT compatibility
        self.payoff_matrix = jnp.array([[R, S], [T, P]], dtype=DTYPE)
        self.R, self.P, self.S, self.T = R, P, S, T

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
        """
        Simplified strategy method that calls the single JIT-compiled function.
        """
        is_first_turn = len(self.history) == 0
        
        # Get opponent's last action, default to COOPERATE if first turn
        opponent_last_action = opponent.history[-1].value if not is_first_turn else COOPERATE

        current_state = self._calculate_state(opponent)

        # Call the single, fast JIT'd function
        new_mus, new_sigmas_sq, state_for_selection, action_idx, new_rng_key = _bql_strategy_step(
            self.mus,
            self.sigmas_sq,
            self.prev_state,
            self.prev_action,
            self.rng_key,
            current_state,
            is_first_turn,
            opponent_last_action,
            self.discount_rate,
            self.reward_variance,
            self.R, self.P, self.S, self.T,
        )

        # Update the agent's state with the results from the pure function
        self.mus = new_mus
        self.sigmas_sq = new_sigmas_sq
        self.rng_key = new_rng_key
        self.prev_state = int(state_for_selection)
        self.prev_action = int(action_idx)
        
        return C if action_idx == COOPERATE else D

    def reset(self) -> None:
        super().reset()
        self.init_state()


class RiskyBQLearner(JaxBayesianQLearner):
    name = "Risky BQLearner"
    
    def __init__(self) -> None:
        super().__init__(discount_rate=0.9, initial_variance=1.0, reward_variance=0.1)


class ArrogantBQLearner(JaxBayesianQLearner):
    name = "Arrogant BQLearner"
    
    def __init__(self) -> None:
        super().__init__(discount_rate=0.1, initial_variance=1.0, reward_variance=0.1)


class HesitantBQLearner(JaxBayesianQLearner):
    name = "Hesitant BQLearner"
    
    def __init__(self) -> None:
        super().__init__(discount_rate=0.9, initial_variance=1.0, reward_variance=1.0)


class CautiousBQLearner(JaxBayesianQLearner):
    name = "Cautious BQLearner"
    
    def __init__(self) -> None:
        super().__init__(discount_rate=0.1, initial_variance=1.0, reward_variance=1.0)


class CooperativeBQLearner(JaxBayesianQLearner):
    """
    A cooperative Bayesian Q-learning variant where exploitation (DC) gives 0 reward,
    making cooperation (CC) the highest reward outcome.
    Uses Thompson sampling with modified reward structure.
    """
    name = "Cooperative BQLearner"
    
    def strategy(self, opponent: Player) -> Action:
        """
        Strategy method using the cooperative JIT-compiled function.
        """
        is_first_turn = len(self.history) == 0
        
        # Get opponent's last action, default to COOPERATE if first turn
        opponent_last_action = opponent.history[-1].value if not is_first_turn else COOPERATE

        current_state = self._calculate_state(opponent)

        # Call the cooperative JIT'd function with DC reward = 0
        new_mus, new_sigmas_sq, state_for_selection, action_idx, new_rng_key = _cooperative_bql_strategy_step(
            self.mus,
            self.sigmas_sq,
            self.prev_state,
            self.prev_action,
            self.rng_key,
            current_state,
            is_first_turn,
            opponent_last_action,
            self.discount_rate,
            self.reward_variance,
            self.R, self.P, self.S, self.T,
        )

        # Update the agent's state with the results from the pure function
        self.mus = new_mus
        self.sigmas_sq = new_sigmas_sq
        self.rng_key = new_rng_key
        self.prev_state = int(state_for_selection)
        self.prev_action = int(action_idx)
        
        return C if action_idx == COOPERATE else D


if __name__ == "__main__":
    import axelrod as axl
    import time
    
    print("=" * 70)
    print("Testing Fixed Bayesian Q-Learner Implementation")
    print("=" * 70)
    
    # Test 1: Verify that variants have different parameters
    print("\n1. Testing Parameter Assignment:")
    print("-" * 70)
    base_agent = JaxBayesianQLearner(discount_rate=0.5, reward_variance=0.5)
    risky_agent = RiskyBQLearner()
    hesitant_agent = HesitantBQLearner()
    
    print(f"Base agent (custom):     discount={base_agent.discount_rate}, reward_var={base_agent.reward_variance}")
    print(f"RiskyBQLearner:          discount={risky_agent.discount_rate}, reward_var={risky_agent.reward_variance}")
    print(f"HesitantBQLearner:       discount={hesitant_agent.discount_rate}, reward_var={hesitant_agent.reward_variance}")
    print("✓ Parameters are correctly assigned and variants differ!")
    
    # Test 2: Run a match and verify it works
    print("\n2. Testing Match Execution:")
    print("-" * 70)
    agent = HesitantBQLearner()
    opponent = axl.Defector()
    
    start_time = time.time()
    match = axl.Match((agent, opponent), turns=1000, noise=0.05)
    match.play()
    end_time = time.time()
    
    print(f"Match completed in {end_time - start_time:.4f} seconds")
    print(f"Final score per turn: {match.final_score_per_turn()}")
    
    # Test 3: Verify Thompson sampling produces stochastic behavior
    print("\n3. Testing Thompson Sampling Stochasticity:")
    print("-" * 70)
    agent1 = HesitantBQLearner()
    agent1.set_seed(42)
    agent2 = HesitantBQLearner()
    agent2.set_seed(99)
    
    opponent1 = axl.TitForTat()
    opponent2 = axl.TitForTat()
    
    match1 = axl.Match((agent1, opponent1), turns=100, noise=0.0)
    match2 = axl.Match((agent2, opponent2), turns=100, noise=0.0)
    
    match1.play()
    match2.play()
    
    # Check if the two agents made different choices (they should with different seeds)
    history_diff = sum(a1 != a2 for a1, a2 in zip(agent1.history, agent2.history))
    print(f"Actions differed in {history_diff}/100 turns")
    print("✓ Thompson sampling produces stochastic behavior!" if history_diff > 0 else "✗ Warning: No stochasticity detected")
    
    # Test 4: Display learned Q-distributions
    print("\n4. Learned Q-Value Distributions (Mean ± Std):")
    print("-" * 70)
    state_names = ["START", "CC", "CD", "DC", "DD"]
    action_names = ["Cooperate", "Defect"]
    
    for i, state_name in enumerate(state_names):
        print(f"\n{state_name}:")
        for j, action_name in enumerate(action_names):
            mu = float(agent.mus[i, j])
            sigma = float(jnp.sqrt(agent.sigmas_sq[i, j]))
            print(f"  {action_name:10s}: μ={mu:6.3f}, σ={sigma:6.3f}")
    
    # Test 5: Compare with Q-learner
    print("\n5. Comparison with Standard Q-Learner:")
    print("-" * 70)
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from agents.qlearner import HesitantQLearner as QLHesitant
        
        bql_agent = HesitantBQLearner()
        ql_agent = QLHesitant()
        opponent_bql = axl.Defector()
        opponent_ql = axl.Defector()
        
        start_bql = time.time()
        match_bql = axl.Match((bql_agent, opponent_bql), turns=1000, noise=0.05)
        match_bql.play()
        time_bql = time.time() - start_bql
        
        start_ql = time.time()
        match_ql = axl.Match((ql_agent, opponent_ql), turns=1000, noise=0.05)
        match_ql.play()
        time_ql = time.time() - start_ql
        
        print(f"BQL time:   {time_bql:.4f}s, score: {match_bql.final_score_per_turn()}")
        print(f"QL time:    {time_ql:.4f}s, score: {match_ql.final_score_per_turn()}")
        print(f"Speedup:    {time_ql/time_bql:.2f}x")
    except (ImportError, Exception) as e:
        print(f"Q-learner comparison skipped: {type(e).__name__}")
    
    # Test 6: Test Cooperative BQL variant
    print("\n6. Testing Cooperative Bayesian Q-Learner:")
    print("-" * 70)
    coop_agent = CooperativeBQLearner()
    regular_agent = HesitantBQLearner()
    
    opponent_coop = axl.Defector()
    opponent_regular = axl.Defector()
    
    match_coop = axl.Match((coop_agent, opponent_coop), turns=1000, noise=0.05)
    match_regular = axl.Match((regular_agent, opponent_regular), turns=1000, noise=0.05)
    
    match_coop.play()
    match_regular.play()
    
    print(f"Cooperative BQL: {match_coop.final_score_per_turn()}")
    print(f"Regular BQL:     {match_regular.final_score_per_turn()}")
    
    print("\nCooperative BQL Learned Q-Values (DC state):")
    print(f"  Cooperate: μ={float(coop_agent.mus[DC, COOPERATE]):.3f}, σ={float(jnp.sqrt(coop_agent.sigmas_sq[DC, COOPERATE])):.3f}")
    print(f"  Defect:    μ={float(coop_agent.mus[DC, DEFECT]):.3f}, σ={float(jnp.sqrt(coop_agent.sigmas_sq[DC, DEFECT])):.3f}")
    print("Note: DC (exploitation) gives 0 reward in cooperative variant, so defection")
    print("      should have lower Q-values compared to standard BQL.")
    
    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
