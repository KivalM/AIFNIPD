"""Game simulator for IPD games."""

from typing import Generator, Optional, Tuple, Dict, Any
from axelrod import Match, Player, Action
import axelrod as axl
import random

# Handle imports - try relative first, then absolute
try:
    from .agent_wrapper import AIFAgentStateWrapper
    from .utils import action_pair_to_outcome, get_payoff
except (ImportError, ValueError):
    # Fallback for when loaded via importlib
    import sys
    import os
    from pathlib import Path
    # Get the directory containing this file
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from agent_wrapper import AIFAgentStateWrapper
    from utils import action_pair_to_outcome, get_payoff


class StepByStepSimulator:
    """Simulator that supports step-by-step execution."""
    
    def __init__(
        self,
        agent: Player,
        opponent: Player,
        turns: int,
        noise: float = 0.0,
        agent_wrapper: Optional[AIFAgentStateWrapper] = None,
    ):
        """Initialize step-by-step simulator."""
        self.agent = agent
        self.opponent = opponent
        self.turns = turns
        self.noise = noise
        self.agent_wrapper = agent_wrapper
        self.current_turn = 0
        self.game_history = []
        self.my_score = 0.0
        self.opponent_score = 0.0
        self.rng = random.Random(42)
        self._initialized = False
        
    def initialize(self):
        """Initialize the simulation (set up match attributes, etc.)."""
        if self._initialized:
            return
        
        # Create match to set up match attributes
        match = Match((self.agent, self.opponent), turns=self.turns, noise=self.noise)
        
        # Set up match attributes on players
        game = getattr(match, 'game', axl.Game())
        
        if not hasattr(self.agent, 'match_attributes'):
            self.agent.match_attributes = {
                "game": game,
                "noise": self.noise,
                "turns": self.turns,
            }
        if not hasattr(self.opponent, 'match_attributes'):
            self.opponent.match_attributes = {
                "game": game,
                "noise": self.noise,
                "turns": self.turns,
            }
        
        # Call receive_match_attributes on both players
        if hasattr(self.agent, 'receive_match_attributes'):
            self.agent.receive_match_attributes()
        if hasattr(self.opponent, 'receive_match_attributes'):
            self.opponent.receive_match_attributes()
        
        # Initialize wrapper if agent is AIF and wrapper not provided
        if self.agent_wrapper is None and hasattr(self.agent, 'agent'):
            self.agent_wrapper = AIFAgentStateWrapper(self.agent)
        
        # Get payoff matrix
        if hasattr(self.agent, 'payoff_matrix'):
            self.payoff_matrix = self.agent.payoff_matrix
        else:
            # Default IPD payoffs
            self.payoff_matrix = {
                Action.C: {Action.C: 3.0, Action.D: 0.0},
                Action.D: {Action.C: 5.0, Action.D: 1.0},
            }
        
        self._initialized = True
    
    def step(self) -> Optional[Dict[str, Any]]:
        """Execute one step of the simulation."""
        if not self._initialized:
            self.initialize()
        
        if self.current_turn >= self.turns:
            return None
        
        # Get current state before action
        if self.current_turn == 0:
            current_state = (None, None)
        else:
            my_last = self.agent.history[-1] if self.agent.history else None
            opp_last = self.opponent.history[-1] if self.opponent.history else None
            current_state = (my_last, opp_last)
        
        # Get agent state if wrapper available (before action)
        agent_state_before = None
        if self.agent_wrapper:
            agent_state_before = self.agent_wrapper.get_current_state(self.current_turn, current_state)
        
        # Play one turn - get intended actions
        my_intended_action = self.agent.strategy(self.opponent)
        opponent_intended_action = self.opponent.strategy(self.agent)
        
        # Apply noise if specified
        my_action = my_intended_action
        opponent_action = opponent_intended_action
        if self.noise > 0:
            if self.rng.random() < self.noise:
                my_action = Action.D if my_action == Action.C else Action.C
            if self.rng.random() < self.noise:
                opponent_action = Action.D if opponent_action == Action.C else Action.C
        
        # Update histories
        self.agent.update_history(my_action, opponent_action)
        self.opponent.update_history(opponent_action, my_action)
        
        # Calculate payoffs
        my_payoff, opponent_payoff = get_payoff(my_action, opponent_action, self.payoff_matrix)
        self.my_score += my_payoff
        self.opponent_score += opponent_payoff
        
        # Record turn data
        turn_data = {
            'turn': self.current_turn + 1,
            'my_action': my_action,
            'opponent_action': opponent_action,
            'my_payoff': my_payoff,
            'opponent_payoff': opponent_payoff,
            'my_score': self.my_score,
            'opponent_score': self.opponent_score,
            'outcome': action_pair_to_outcome(my_action, opponent_action),
        }
        self.game_history.append(turn_data)
        
        # Get final agent state after this turn
        agent_state_after = None
        if self.agent_wrapper:
            agent_state_after = self.agent_wrapper.get_current_state(
                self.current_turn + 1, 
                (my_action, opponent_action)
            )
        else:
            agent_state_after = agent_state_before
        
        self.current_turn += 1
        
        return {
            'turn': self.current_turn,
            'total_turns': self.turns,
            'history': self.game_history.copy(),
            'my_score': self.my_score,
            'opponent_score': self.opponent_score,
            'agent_state': agent_state_after,
            'is_complete': self.current_turn >= self.turns,
        }
    
    def reset(self):
        """Reset the simulator."""
        self.current_turn = 0
        self.game_history = []
        self.my_score = 0.0
        self.opponent_score = 0.0
        self.rng = random.Random(42)
        self._initialized = False
        if self.agent_wrapper:
            self.agent_wrapper.clear_cache()
        self.agent.reset()
        if hasattr(self.opponent, 'reset'):
            self.opponent.reset()


def simulate_game(
    agent: Player,
    opponent: Player,
    turns: int,
    noise: float = 0.0,
    update_interval: int = 1,
    agent_wrapper: Optional[AIFAgentStateWrapper] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Simulate an IPD game and yield state after each turn/batch.
    
    Args:
        agent: The main agent to simulate
        opponent: The opponent agent
        turns: Number of turns to play
        noise: Noise level (probability of action flip)
        update_interval: Yield state every N turns
        agent_wrapper: Optional wrapper for extracting agent state
        
    Yields:
        Dictionary with turn information and agent state
    """
    # Create match to set up match attributes
    match = Match((agent, opponent), turns=turns, noise=noise)
    
    # Set up match attributes on players (Match does this internally, but we need to do it manually)
    # since we're not using match.play()
    # Use the match's game if available, otherwise create a default one
    game = getattr(match, 'game', axl.Game())
    
    if not hasattr(agent, 'match_attributes'):
        agent.match_attributes = {
            "game": game,
            "noise": noise,
            "turns": turns,
        }
    if not hasattr(opponent, 'match_attributes'):
        opponent.match_attributes = {
            "game": game,
            "noise": noise,
            "turns": turns,
        }
    
    # Call receive_match_attributes on both players
    if hasattr(agent, 'receive_match_attributes'):
        agent.receive_match_attributes()
    if hasattr(opponent, 'receive_match_attributes'):
        opponent.receive_match_attributes()
    
    # Initialize wrapper if agent is AIF and wrapper not provided
    if agent_wrapper is None and hasattr(agent, 'agent'):
        agent_wrapper = AIFAgentStateWrapper(agent)
    
    # Track history
    game_history = []
    my_score = 0.0
    opponent_score = 0.0
    
    # Get payoff matrix
    if hasattr(agent, 'payoff_matrix'):
        payoff_matrix = agent.payoff_matrix
    else:
        # Default IPD payoffs
        payoff_matrix = {
            Action.C: {Action.C: 3.0, Action.D: 0.0},
            Action.D: {Action.C: 5.0, Action.D: 1.0},
        }
    
    # Random number generator for noise
    rng = random.Random(42)
    
    # Play game turn by turn
    for turn in range(turns):
        # Get current state before action
        if turn == 0:
            current_state = (None, None)
        else:
            my_last = agent.history[-1] if agent.history else None
            opp_last = opponent.history[-1] if opponent.history else None
            current_state = (my_last, opp_last)
        
        # Get agent state if wrapper available
        agent_state = None
        if agent_wrapper:
            agent_state = agent_wrapper.get_current_state(turn, current_state)
        
        # Play one turn - get intended actions
        my_intended_action = agent.strategy(opponent)
        opponent_intended_action = opponent.strategy(agent)
        
        # Apply noise if specified
        my_action = my_intended_action
        opponent_action = opponent_intended_action
        if noise > 0:
            if rng.random() < noise:
                my_action = Action.D if my_action == Action.C else Action.C
            if rng.random() < noise:
                opponent_action = Action.D if opponent_action == Action.C else Action.C
        
        # Update histories
        agent.update_history(my_action, opponent_action)
        opponent.update_history(opponent_action, my_action)
        
        # Calculate payoffs
        my_payoff, opponent_payoff = get_payoff(my_action, opponent_action, payoff_matrix)
        my_score += my_payoff
        opponent_score += opponent_payoff
        
        # Record turn data
        turn_data = {
            'turn': turn + 1,
            'my_action': my_action,
            'opponent_action': opponent_action,
            'my_payoff': my_payoff,
            'opponent_payoff': opponent_payoff,
            'my_score': my_score,
            'opponent_score': opponent_score,
            'outcome': action_pair_to_outcome(my_action, opponent_action),
        }
        game_history.append(turn_data)
        
        # Yield state at specified intervals
        if (turn + 1) % update_interval == 0 or turn == turns - 1:
            # Get final agent state after this turn
            if agent_wrapper:
                final_agent_state = agent_wrapper.get_current_state(turn + 1, (my_action, opponent_action))
            else:
                final_agent_state = agent_state
            
            yield {
                'turn': turn + 1,
                'total_turns': turns,
                'history': game_history.copy(),
                'my_score': my_score,
                'opponent_score': opponent_score,
                'agent_state': final_agent_state,
                'match': match,
            }

