from typing import Dict, Tuple
import numpy as np
from axelrod.action import Action
from axelrod.player import Player

# Define actions and states as constants
C, D = Action.C, Action.D
COOPERATE, DEFECT = 0, 1
START, CC, CD, DC, DD = 0, 1, 2, 3, 4


class DynaQ(Player):
    """
    Dyna-Q agent that combines Q-learning with model-based planning.
    
    After each real experience, the agent performs N planning steps where it:
    1. Samples a random previously visited state-action pair
    2. Uses the learned model to predict the next state and reward
    3. Updates Q-values based on this simulated experience
    
    This allows the agent to learn more efficiently from limited experience.
    """
    
    name = "Dyna-Q"
    learning_rate = 0.5
    discount_rate = 0.9
    action_selection_parameter = 0.1  # epsilon for epsilon-greedy
    planning_steps = 10  # Number of planning steps per real experience
    
    def __init__(
        self,
        learning_rate: float = 0.5,
        discount_rate: float = 0.9,
        action_selection_parameter: float = 0.1,
        planning_steps: int = 10
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.action_selection_parameter = action_selection_parameter
        self.planning_steps = planning_steps
        self.classifier["stochastic"] = True
        self.set_seed(0)
        self.init_state()
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def init_state(self):
        """Initialize or reset the agent's state, Q-table, and model."""
        # Q-table: 5 states × 2 actions
        self.Qs = np.zeros((5, 2), dtype=np.float32)
        # V-values: maximum Q-value for each state
        self.Vs = np.zeros(5, dtype=np.float32)
        # Previous state and action
        self.prev_state = START
        self.prev_action = COOPERATE
        
        # Model: stores (next_state, reward) for each state-action pair
        # Model[state][action] = (next_state, reward)
        self.model: Dict[Tuple[int, int], Tuple[int, float]] = {}
        
        # Keep track of visited state-action pairs for planning
        self.visited_state_actions = []
    
    def receive_match_attributes(self):
        """Receive game payoff matrix."""
        (self.R, self.P, self.S, self.T) = self.match_attributes["game"].RPST()
        # Create payoff matrix [my_action, opponent_action] -> reward
        self.payoff_matrix = np.array([[self.R, self.S], [self.T, self.P]], dtype=np.float32)
    
    def q_learning_update(self, state: int, action: int, reward: float, next_state: int):
        """Perform Q-learning update."""
        # Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]
        td_target = reward + self.discount_rate * self.Vs[next_state]
        self.Qs[state, action] += self.learning_rate * (td_target - self.Qs[state, action])
        # Update V-value for the state
        self.Vs[state] = np.max(self.Qs[state])
    
    def update_model(self, state: int, action: int, next_state: int, reward: float):
        """Update the learned model with observed transition."""
        key = (state, action)
        if key not in self.model:
            self.visited_state_actions.append(key)
        self.model[key] = (next_state, reward)
    
    def planning(self):
        """Perform planning steps using the learned model."""
        if len(self.visited_state_actions) == 0:
            return
        
        for _ in range(self.planning_steps):
            # Sample a random previously visited state-action pair
            state, action = self.visited_state_actions[
                self.rng.randint(len(self.visited_state_actions))
            ]
            
            # Get predicted next state and reward from model
            next_state, reward = self.model[(state, action)]
            
            # Perform Q-learning update using simulated experience
            self.q_learning_update(state, action, reward, next_state)
    
    def select_action(self, state: int) -> int:
        """Select action using epsilon-greedy policy."""
        if self.rng.random() < self.action_selection_parameter:
            # Explore: random action
            return self.rng.choice([COOPERATE, DEFECT])
        else:
            # Exploit: best action
            return int(np.argmax(self.Qs[state]))
    
    def strategy(self, opponent: Player) -> Action:
        """Main strategy method called each turn."""
        is_first_turn = len(self.history) == 0
        
        if is_first_turn:
            # First turn: select action from START state
            current_state = START
            action = self.select_action(current_state)
            
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
        
        # (a) Q-learning update from real experience
        self.q_learning_update(self.prev_state, self.prev_action, reward, current_state)
        
        # (b) Model learning: store the transition
        self.update_model(self.prev_state, self.prev_action, current_state, reward)
        
        # (c) Planning: simulate experiences and learn from them
        self.planning()
        
        # (d) Select next action
        action = self.select_action(current_state)
        
        # Update state for next turn
        self.prev_state = current_state
        self.prev_action = action
        
        return C if action == COOPERATE else D
    
    def reset(self) -> None:
        """Reset the agent for a new match."""
        super().reset()
        self.init_state()


class DynaQPlus(DynaQ):
    """
    Dyna-Q+ variant that encourages exploration of state-action pairs
    that haven't been visited recently by adding a bonus to their rewards.
    """
    
    name = "Dyna-Q+"
    exploration_bonus = 0.001  # κ parameter for exploration bonus
    
    def __init__(
        self,
        learning_rate: float = 0.5,
        discount_rate: float = 0.9,
        action_selection_parameter: float = 0.1,
        planning_steps: int = 10,
        exploration_bonus: float = 0.001
    ) -> None:
        self.exploration_bonus = exploration_bonus
        super().__init__(learning_rate, discount_rate, action_selection_parameter, planning_steps)
    
    def init_state(self):
        """Initialize state including time since last visit."""
        super().init_state()
        # Track time steps since each state-action pair was last visited
        self.time_since_visit = {}
        self.current_time = 0
    
    def update_model(self, state: int, action: int, next_state: int, reward: float):
        """Update model and track visit times."""
        super().update_model(state, action, next_state, reward)
        self.time_since_visit[(state, action)] = self.current_time
        self.current_time += 1
    
    def planning(self):
        """Planning with exploration bonus for long-unvisited state-actions."""
        if len(self.visited_state_actions) == 0:
            return
        
        for _ in range(self.planning_steps):
            # Sample a random previously visited state-action pair
            state, action = self.visited_state_actions[
                self.rng.randint(len(self.visited_state_actions))
            ]
            
            # Get predicted next state and reward from model
            next_state, reward = self.model[(state, action)]
            
            # Add exploration bonus based on time since last visit
            time_delta = self.current_time - self.time_since_visit[(state, action)]
            bonus = self.exploration_bonus * np.sqrt(time_delta)
            augmented_reward = reward + bonus
            
            # Perform Q-learning update with augmented reward
            self.q_learning_update(state, action, augmented_reward, next_state)


# Variants with different hyperparameters
class RiskyDynaQ(DynaQ):
    """Dyna-Q with high learning rate."""
    name = "Risky Dyna-Q"
    learning_rate = 0.9
    discount_rate = 0.9
    planning_steps = 20


class CautiousDynaQ(DynaQ):
    """Dyna-Q with low learning rate and more planning."""
    name = "Cautious Dyna-Q"
    learning_rate = 0.1
    discount_rate = 0.1
    planning_steps = 50


class CooperativeDynaQ(DynaQ):
    """
    Cooperative Dyna-Q variant where exploitation (DC) gives 0 reward,
    making cooperation (CC) the highest reward outcome.
    """
    name = "Cooperative Dyna-Q"
    
    def receive_match_attributes(self):
        """Receive game payoff matrix with modified DC reward."""
        (self.R, self.P, self.S, self.T) = self.match_attributes["game"].RPST()
        # Create payoff matrix with DC reward set to 0
        self.payoff_matrix = np.array([[self.R, self.S], [0, self.P]], dtype=np.float32)


if __name__ == "__main__":
    import axelrod as axl
    import time
    
    print("Testing Dyna-Q agents...\n")
    
    # Test standard Dyna-Q
    agent = DynaQ(planning_steps=10)
    opponent = axl.TitForTat()
    
    start_time = time.time()
    match = axl.Match((agent, opponent), turns=1000)
    match.play()
    end_time = time.time()
    
    print(f"Dyna-Q vs Tit For Tat:")
    print(f"  Time: {end_time - start_time:.4f} seconds")
    print(f"  Final score per turn: {match.final_score_per_turn()}")
    print(f"  Cooperation rate: {agent.history.cooperations / len(agent.history):.2%}")
    print(f"  Q-values at CC state: {agent.Qs[CC]}")
    print()
    
    # Test Dyna-Q+
    agent_plus = DynaQPlus(planning_steps=10)
    opponent_plus = axl.TitForTat()
    
    start_time = time.time()
    match_plus = axl.Match((agent_plus, opponent_plus), turns=1000)
    match_plus.play()
    end_time = time.time()
    
    print(f"Dyna-Q+ vs Tit For Tat:")
    print(f"  Time: {end_time - start_time:.4f} seconds")
    print(f"  Final score per turn: {match_plus.final_score_per_turn()}")
    print(f"  Cooperation rate: {agent_plus.history.cooperations / len(agent_plus.history):.2%}")
    print(f"  Q-values at CC state: {agent_plus.Qs[CC]}")
    print()
    
    # Test against a more challenging opponent
    agent_risky = RiskyDynaQ()
    opponent_defector = axl.Defector()
    
    start_time = time.time()
    match_risky = axl.Match((agent_risky, opponent_defector), turns=1000)
    match_risky.play()
    end_time = time.time()
    
    print(f"Risky Dyna-Q vs Defector:")
    print(f"  Time: {end_time - start_time:.4f} seconds")
    print(f"  Final score per turn: {match_risky.final_score_per_turn()}")
    print(f"  Cooperation rate: {agent_risky.history.cooperations / len(agent_risky.history):.2%}")
    print(f"  Q-values at CD state: {agent_risky.Qs[CD]}")
    print()
    
    # Compare Q-learner vs Dyna-Q learning speed
    print("Comparing learning efficiency (short matches):")
    print("-" * 50)
    
    from qlearner import JaxQLearner
    
    for turns in [50, 100, 200]:
        # Q-learner
        q_agent = JaxQLearner()
        q_opp = axl.TitForTat()
        q_match = axl.Match((q_agent, q_opp), turns=turns)
        q_match.play()
        q_score = q_match.final_score_per_turn()
        
        # Dyna-Q
        dyna_agent = DynaQ()
        dyna_opp = axl.TitForTat()
        dyna_match = axl.Match((dyna_agent, dyna_opp), turns=turns)
        dyna_match.play()
        dyna_score = dyna_match.final_score_per_turn()
        
        print(f"{turns} turns:")
        print(f"  Q-learner score: {q_score[0]:.3f}")
        print(f"  Dyna-Q score: {dyna_score[0]:.3f}")
        print(f"  Improvement: {((dyna_score[0] - q_score[0]) / q_score[0] * 100):.1f}%")
        print()

