from typing import Tuple
import numpy as np
from scipy.stats import dirichlet
from axelrod.action import Action
from axelrod.player import Player

# Define actions and states as constants
C, D = Action.C, Action.D
COOPERATE, DEFECT = 0, 1
START, CC, CD, DC, DD = 0, 1, 2, 3, 4


class PSRL(Player):
    """
    Posterior Sampling for Reinforcement Learning (PSRL) agent for IPD.
    
    Uses Bayesian model-based RL with Thompson sampling:
    1. Maintains Dirichlet posteriors over transition probabilities
    2. Tracks empirical reward estimates for each state-action pair
    3. Each turn: samples a model, computes optimal policy via value iteration, acts
    4. Updates posterior beliefs after observing transitions and rewards
    
    This is the most Bayesian approach - re-sampling and re-planning every turn.
    """
    
    name = "PSRL"
    prior_strength = 1.0  # Initial Dirichlet concentration parameter
    value_iteration_steps = 100  # Maximum iterations for policy computation
    convergence_threshold = 1e-4  # Convergence threshold for value iteration
    discount_rate = 0.9  # Discount factor for value iteration
    
    def __init__(
        self,
        prior_strength: float = 1.0,
        value_iteration_steps: int = 100,
        convergence_threshold: float = 1e-4,
        discount_rate: float = 0.9
    ) -> None:
        super().__init__()
        self.prior_strength = prior_strength
        self.value_iteration_steps = value_iteration_steps
        self.convergence_threshold = convergence_threshold
        self.discount_rate = discount_rate
        self.classifier["stochastic"] = True
        self.set_seed(0)
        self.init_state()
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def init_state(self):
        """Initialize or reset the agent's state, priors, and tracking."""
        # Dirichlet priors for transitions: transition_counts[state][action][next_state]
        # Initialize with uniform prior (all concentrations = prior_strength)
        self.transition_counts = np.full((5, 2, 5), self.prior_strength, dtype=np.float32)
        
        # Reward tracking: empirical sums and counts
        self.reward_sums = np.zeros((5, 2), dtype=np.float32)
        self.reward_counts = np.zeros((5, 2), dtype=np.float32)
        
        # Previous state and action for updates
        self.prev_state = START
        self.prev_action = COOPERATE
    
    def receive_match_attributes(self):
        """Receive game payoff matrix."""
        (self.R, self.P, self.S, self.T) = self.match_attributes["game"].RPST()
        # Create payoff matrix [my_action, opponent_action] -> reward
        self.payoff_matrix = np.array([[self.R, self.S], [self.T, self.P]], dtype=np.float32)
    
    def sample_model(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a plausible MDP model from the current posterior.
        
        Returns:
            transition_probs: [5, 2, 5] array of sampled transition probabilities
            expected_rewards: [5, 2] array of expected rewards
        """
        transition_probs = np.zeros((5, 2, 5), dtype=np.float32)
        
        # Sample transition probabilities from Dirichlet posterior for each state-action pair
        for state in range(5):
            for action in range(2):
                # Dirichlet parameters are the transition counts
                alpha = self.transition_counts[state, action, :]
                # Sample next-state distribution
                transition_probs[state, action, :] = dirichlet.rvs(alpha, size=1, random_state=self.rng)[0]
        
        # Estimate expected rewards as empirical means
        expected_rewards = np.zeros((5, 2), dtype=np.float32)
        for state in range(5):
            for action in range(2):
                count = self.reward_counts[state, action]
                if count > 0:
                    expected_rewards[state, action] = self.reward_sums[state, action] / count
                else:
                    # No observations yet, use neutral estimate
                    expected_rewards[state, action] = 0.0
        
        return transition_probs, expected_rewards
    
    def value_iteration(self, transition_probs: np.ndarray, expected_rewards: np.ndarray) -> np.ndarray:
        """
        Compute optimal policy for a given MDP model using value iteration.
        
        Args:
            transition_probs: [5, 2, 5] transition probability matrix
            expected_rewards: [5, 2] expected reward matrix
        
        Returns:
            policy: [5] array mapping each state to optimal action
        """
        # Initialize value function
        V = np.zeros(5, dtype=np.float32)
        
        # Perform value iteration
        for iteration in range(self.value_iteration_steps):
            V_old = V.copy()
            
            # Update value for each state
            for state in range(5):
                # Compute Q-values for both actions
                Q_values = np.zeros(2, dtype=np.float32)
                for action in range(2):
                    # Q(s,a) = R(s,a) + γ * Σ P(s'|s,a) * V(s')
                    immediate_reward = expected_rewards[state, action]
                    expected_future_value = np.dot(transition_probs[state, action, :], V_old)
                    Q_values[action] = immediate_reward + self.discount_rate * expected_future_value
                
                # Update value as max over actions
                V[state] = np.max(Q_values)
            
            # Check convergence
            if np.max(np.abs(V - V_old)) < self.convergence_threshold:
                break
        
        # Extract greedy policy from final value function
        policy = np.zeros(5, dtype=np.int32)
        for state in range(5):
            Q_values = np.zeros(2, dtype=np.float32)
            for action in range(2):
                immediate_reward = expected_rewards[state, action]
                expected_future_value = np.dot(transition_probs[state, action, :], V)
                Q_values[action] = immediate_reward + self.discount_rate * expected_future_value
            policy[state] = np.argmax(Q_values)
        
        return policy
    
    def update_beliefs(self, state: int, action: int, next_state: int, reward: float):
        """
        Update posterior beliefs based on observed transition and reward.
        
        Args:
            state: Previous state
            action: Action taken
            next_state: Observed next state
            reward: Observed reward
        """
        # Update transition counts (Dirichlet posterior)
        self.transition_counts[state, action, next_state] += 1.0
        
        # Update reward statistics
        self.reward_sums[state, action] += reward
        self.reward_counts[state, action] += 1.0
    
    def strategy(self, opponent: Player) -> Action:
        """
        Main strategy method: Sample model → Plan → Act → Update.
        """
        is_first_turn = len(self.history) == 0
        
        if is_first_turn:
            # First turn: sample model and plan
            transition_probs, expected_rewards = self.sample_model()
            policy = self.value_iteration(transition_probs, expected_rewards)
            
            # Select action from START state
            current_state = START
            action = int(policy[current_state])
            
            # Update state for next turn
            self.prev_state = current_state
            self.prev_action = action
            
            return C if action == COOPERATE else D
        
        # Get opponent's last action
        opponent_last_action = opponent.history[-1].value
        
        # Determine current state based on last turn
        # State encoding: START=0, CC=1, CD=2, DC=3, DD=4
        current_state = 1 + (self.prev_action * 2) + opponent_last_action
        
        # Get reward from last turn
        reward = self.payoff_matrix[self.prev_action, opponent_last_action]
        
        # Update beliefs with observed transition and reward
        self.update_beliefs(self.prev_state, self.prev_action, current_state, reward)
        
        # Sample new model and compute optimal policy
        transition_probs, expected_rewards = self.sample_model()
        policy = self.value_iteration(transition_probs, expected_rewards)
        
        # Select action according to policy
        action = int(policy[current_state])
        
        # Update state for next turn
        self.prev_state = current_state
        self.prev_action = action
        
        return C if action == COOPERATE else D
    
    def reset(self) -> None:
        """Reset the agent for a new match."""
        super().reset()
        self.init_state()


class OptimisticPSRL(PSRL):
    """
    Optimistic PSRL variant with higher priors for cooperative outcomes.
    Encourages exploration of cooperative strategies.
    """
    
    name = "Optimistic PSRL"
    
    def init_state(self):
        """Initialize with optimistic priors favoring cooperation."""
        super().init_state()
        
        # Boost priors for transitions that lead to cooperative states
        # From START: favor cooperate
        self.transition_counts[START, COOPERATE, :] *= 2.0
        
        # From CC: favor staying in CC (both cooperate)
        self.transition_counts[CC, COOPERATE, CC] *= 3.0
        
        # From CD: favor returning to cooperation
        self.transition_counts[CD, COOPERATE, CC] *= 2.0
        
        # From DC: favor forgiving (cooperate)
        self.transition_counts[DC, COOPERATE, CC] *= 2.0


class PessimisticPSRL(PSRL):
    """
    Pessimistic PSRL variant with stronger uniform priors.
    More skeptical about the opponent's cooperation.
    """
    
    name = "Pessimistic PSRL"
    prior_strength = 5.0  # Stronger prior requires more evidence to update beliefs


class FastPSRL(PSRL):
    """
    Fast PSRL variant with fewer value iteration steps.
    Trades off optimality for computational speed.
    """
    
    name = "Fast PSRL"
    value_iteration_steps = 20  # Fewer iterations for faster computation


class CooperativePSRL(PSRL):
    """
    Cooperative PSRL variant where exploitation (DC) gives 0 reward,
    making cooperation (CC) the highest reward outcome.
    """
    name = "Cooperative PSRL"
    
    def receive_match_attributes(self):
        """Receive game payoff matrix with modified DC reward."""
        (self.R, self.P, self.S, self.T) = self.match_attributes["game"].RPST()
        # Create payoff matrix with DC reward set to 0
        self.payoff_matrix = np.array([[self.R, self.S], [0, self.P]], dtype=np.float32)


if __name__ == "__main__":
    import axelrod as axl
    import time
    
    print("Testing PSRL agents...\n")
    
    # Test standard PSRL
    print("=" * 60)
    print("PSRL vs Tit For Tat")
    print("=" * 60)
    agent = PSRL()
    opponent = axl.TitForTat()
    
    start_time = time.time()
    match = axl.Match((agent, opponent), turns=200)
    match.play()
    end_time = time.time()
    
    print(f"Time: {end_time - start_time:.4f} seconds")
    print(f"Final score per turn: {match.final_score_per_turn()}")
    print(f"Cooperation rate: {agent.history.cooperations / len(agent.history):.2%}")
    print(f"Transition counts for CC state (cooperate): {agent.transition_counts[CC, COOPERATE, :]}")
    print()
    
    # Test against Defector
    print("=" * 60)
    print("PSRL vs Defector")
    print("=" * 60)
    agent_vs_def = PSRL()
    opponent_def = axl.Defector()
    
    start_time = time.time()
    match_def = axl.Match((agent_vs_def, opponent_def), turns=200)
    match_def.play()
    end_time = time.time()
    
    print(f"Time: {end_time - start_time:.4f} seconds")
    print(f"Final score per turn: {match_def.final_score_per_turn()}")
    print(f"Cooperation rate: {agent_vs_def.history.cooperations / len(agent_vs_def.history):.2%}")
    print(f"Transition counts for CD state (cooperate): {agent_vs_def.transition_counts[CD, COOPERATE, :]}")
    print(f"Transition counts for CD state (defect): {agent_vs_def.transition_counts[CD, DEFECT, :]}")
    print()
    
    # Test Optimistic PSRL
    print("=" * 60)
    print("Optimistic PSRL vs Tit For Tat")
    print("=" * 60)
    agent_opt = OptimisticPSRL()
    opponent_opt = axl.TitForTat()
    
    start_time = time.time()
    match_opt = axl.Match((agent_opt, opponent_opt), turns=200)
    match_opt.play()
    end_time = time.time()
    
    print(f"Time: {end_time - start_time:.4f} seconds")
    print(f"Final score per turn: {match_opt.final_score_per_turn()}")
    print(f"Cooperation rate: {agent_opt.history.cooperations / len(agent_opt.history):.2%}")
    print()
    
    # Test Fast PSRL
    print("=" * 60)
    print("Fast PSRL vs Tit For Tat (Speed Test)")
    print("=" * 60)
    agent_fast = FastPSRL()
    opponent_fast = axl.TitForTat()
    
    start_time = time.time()
    match_fast = axl.Match((agent_fast, opponent_fast), turns=200)
    match_fast.play()
    end_time = time.time()
    
    print(f"Time: {end_time - start_time:.4f} seconds")
    print(f"Final score per turn: {match_fast.final_score_per_turn()}")
    print(f"Cooperation rate: {agent_fast.history.cooperations / len(agent_fast.history):.2%}")
    print()
    
    # Compare learning efficiency: PSRL vs Q-learner vs Dyna-Q
    print("=" * 60)
    print("Comparing Learning Efficiency (Short Matches)")
    print("=" * 60)
    
    try:
        from qlearner import JaxQLearner
        from dynaQ import DynaQ
    except Exception as e:
        print(f"Warning: Could not import comparison agents: {e}")
        print("Skipping comparison tests...")
        import sys
        sys.exit(0)
    
    for turns in [50, 100, 200]:
        print(f"\n{turns} turns against Tit For Tat:")
        print("-" * 40)
        
        # Q-learner
        q_agent = JaxQLearner()
        q_opp = axl.TitForTat()
        q_match = axl.Match((q_agent, q_opp), turns=turns)
        q_match.play()
        q_score = q_match.final_score_per_turn()
        q_coop = q_agent.history.cooperations / len(q_agent.history)
        
        # Dyna-Q
        dyna_agent = DynaQ()
        dyna_opp = axl.TitForTat()
        dyna_match = axl.Match((dyna_agent, dyna_opp), turns=turns)
        dyna_match.play()
        dyna_score = dyna_match.final_score_per_turn()
        dyna_coop = dyna_agent.history.cooperations / len(dyna_agent.history)
        
        # PSRL
        psrl_agent = PSRL()
        psrl_opp = axl.TitForTat()
        psrl_match = axl.Match((psrl_agent, psrl_opp), turns=turns)
        psrl_match.play()
        psrl_score = psrl_match.final_score_per_turn()
        psrl_coop = psrl_agent.history.cooperations / len(psrl_agent.history)
        
        print(f"  Q-learner:     score={q_score[0]:.3f}, coop={q_coop:.2%}")
        print(f"  Dyna-Q:        score={dyna_score[0]:.3f}, coop={dyna_coop:.2%}")
        print(f"  PSRL:          score={psrl_score[0]:.3f}, coop={psrl_coop:.2%}")
    
    print("\n" + "=" * 60)
    print("Testing belief convergence over time")
    print("=" * 60)
    
    # Show how beliefs converge with more observations
    agent_conv = PSRL()
    opponent_conv = axl.TitForTat()
    
    match_conv = axl.Match((agent_conv, opponent_conv), turns=500)
    match_conv.play()
    
    print("\nFinal transition beliefs for CC state:")
    print(f"  After cooperate action: {agent_conv.transition_counts[CC, COOPERATE, :] / agent_conv.transition_counts[CC, COOPERATE, :].sum()}")
    print(f"  After defect action: {agent_conv.transition_counts[CC, DEFECT, :] / agent_conv.transition_counts[CC, DEFECT, :].sum()}")
    print(f"\nTotal transitions observed from CC: {agent_conv.transition_counts[CC, :, :].sum() - 10}")  # Subtract initial prior
    print(f"Final cooperation rate: {agent_conv.history.cooperations / len(agent_conv.history):.2%}")

