"""Utility functions for the web UI."""

from typing import Tuple, Optional
from axelrod.action import Action
import pandas as pd


def action_pair_to_outcome(my_action: Action, opponent_action: Action) -> str:
    """Convert action pair to outcome string."""
    if my_action == Action.C and opponent_action == Action.C:
        return "CC"
    elif my_action == Action.C and opponent_action == Action.D:
        return "CD"
    elif my_action == Action.D and opponent_action == Action.C:
        return "DC"
    else:
        return "DD"


def get_outcome_color(outcome: str) -> str:
    """Get color for outcome."""
    colors = {
        "CC": "#2ecc71",  # Green
        "CD": "#e74c3c",  # Red
        "DC": "#f39c12",  # Orange
        "DD": "#95a5a6",  # Gray
    }
    return colors.get(outcome, "#000000")


def get_payoff(my_action: Action, opponent_action: Action, payoff_matrix: dict) -> Tuple[float, float]:
    """Get payoffs for both players."""
    my_payoff = payoff_matrix[my_action][opponent_action]
    opponent_payoff = payoff_matrix[opponent_action][my_action]
    return my_payoff, opponent_payoff


def create_history_dataframe(history: list) -> pd.DataFrame:
    """Create a pandas DataFrame from game history."""
    if not history:
        return pd.DataFrame(columns=[
            'Turn', 'My Action', 'Opponent Action', 'Outcome',
            'My Payoff', 'Opponent Payoff', 'My Score', 'Opponent Score'
        ])
    
    rows = []
    my_score = 0.0
    opponent_score = 0.0
    
    for turn_data in history:
        turn = turn_data['turn']
        my_action = turn_data['my_action']
        opponent_action = turn_data['opponent_action']
        outcome = action_pair_to_outcome(my_action, opponent_action)
        my_payoff = turn_data['my_payoff']
        opponent_payoff = turn_data['opponent_payoff']
        
        my_score += my_payoff
        opponent_score += opponent_payoff
        
        rows.append({
            'Turn': turn,
            'My Action': 'C' if my_action == Action.C else 'D',
            'Opponent Action': 'C' if opponent_action == Action.C else 'D',
            'Outcome': outcome,
            'My Payoff': my_payoff,
            'Opponent Payoff': opponent_payoff,
            'My Score': my_score,
            'Opponent Score': opponent_score,
        })
    
    return pd.DataFrame(rows)

