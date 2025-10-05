"""
Utility functions for processing hyperparameter experiment results.
"""
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


def get_noise_levels(results_dir: Path) -> List[float]:
    """
    Extract all noise levels from the results directory.
    
    Args:
        results_dir: Path to the results directory
        
    Returns:
        List of noise levels as floats
    """
    # Look at the first agent/hyperparam/repetition to get noise levels
    for agent_dir in results_dir.iterdir():
        if agent_dir.is_dir() and agent_dir.name not in ['.gitignore']:
            for hyperparam_dir in agent_dir.iterdir():
                if hyperparam_dir.is_dir():
                    for rep_dir in hyperparam_dir.iterdir():
                        if rep_dir.is_dir() and rep_dir.name.startswith('repetition_'):
                            noise_files = sorted([f for f in rep_dir.glob('*.csv')])
                            return sorted([float(f.stem) for f in noise_files])
    return []


def load_agent_score_from_csv(csv_file: Path) -> float:
    """
    Load mean score for an agent from a CSV file.
    
    Args:
        csv_file: Path to the CSV file containing agent results
        
    Returns:
        Mean score for the agent (player index 0)
    """
    df = pd.read_csv(csv_file)
    
    # Get scores for player index 0 (the agent being tested)
    agent_scores = df[df['Player index'] == 0]['Score']
    
    return agent_scores.mean()


def get_agent_mean_scores(
    results_dir: Path,
    agent_type: str,
    hyperparam_combo: str,
    noise_level: float
) -> Tuple[float, int]:
    """
    Calculate the mean score across all repetitions for a given agent configuration.
    
    Args:
        results_dir: Path to the results directory
        agent_type: Agent type directory name (e.g., 'qlearning', 'bqlearning')
        hyperparam_combo: Hyperparameter combination directory name
        noise_level: Noise level to analyze
        
    Returns:
        Tuple of (mean_score, num_repetitions)
    """
    agent_dir = results_dir / agent_type / hyperparam_combo
    scores = []
    
    for rep_dir in sorted(agent_dir.glob('repetition_*')):
        csv_file = rep_dir / f"{noise_level}.csv"
        if csv_file.exists():
            score = load_agent_score_from_csv(csv_file)
            scores.append(score)
    
    if not scores:
        return 0.0, 0
    
    return sum(scores) / len(scores), len(scores)


def get_all_hyperparams(results_dir: Path, agent_type: str) -> List[str]:
    """
    Get all hyperparameter combinations for a given agent type.
    
    Args:
        results_dir: Path to the results directory
        agent_type: Agent type directory name
        
    Returns:
        List of hyperparameter combination directory names
    """
    agent_dir = results_dir / agent_type
    if not agent_dir.exists():
        return []
    
    return sorted([d.name for d in agent_dir.iterdir() if d.is_dir()])


def analyze_best_agents_per_noise(
    results_dir: Path,
    dir_agent_map: Dict[str, str]
) -> Dict[float, pd.DataFrame]:
    """
    Find the best performing agents for each noise level.
    
    Args:
        results_dir: Path to the results directory
        dir_agent_map: Mapping from directory names to display names
        
    Returns:
        Dictionary mapping noise levels to DataFrames with best agent results
    """
    noise_levels = get_noise_levels(results_dir)
    results_by_noise = {}
    
    for noise_level in noise_levels:
        agent_results = []
        
        for agent_dir, agent_display_name in dir_agent_map.items():
            hyperparam_combos = get_all_hyperparams(results_dir, agent_dir)
            
            for hyperparam_combo in hyperparam_combos:
                mean_score, num_reps = get_agent_mean_scores(
                    results_dir, agent_dir, hyperparam_combo, noise_level
                )
                
                if num_reps > 0:
                    agent_results.append({
                        'Agent Type': agent_display_name,
                        'Hyperparameters': hyperparam_combo,
                        'Mean Score': mean_score,
                        'Num Repetitions': num_reps
                    })
        
        # Create DataFrame and sort by mean score
        df = pd.DataFrame(agent_results)
        df = df.sort_values('Mean Score', ascending=False).reset_index(drop=True)
        results_by_noise[noise_level] = df
    
    return results_by_noise


def get_top_n_agents(
    results_by_noise: Dict[float, pd.DataFrame],
    n: int = 10
) -> Dict[float, pd.DataFrame]:
    """
    Get the top N agents for each noise level.
    
    Args:
        results_by_noise: Dictionary mapping noise levels to DataFrames
        n: Number of top agents to return
        
    Returns:
        Dictionary mapping noise levels to DataFrames with top N agents
    """
    return {noise: df.head(n) for noise, df in results_by_noise.items()}


def format_results_for_display(
    results_by_noise: Dict[float, pd.DataFrame],
    decimals: int = 2
) -> Dict[float, pd.DataFrame]:
    """
    Format results for display by rounding scores.
    
    Args:
        results_by_noise: Dictionary mapping noise levels to DataFrames
        decimals: Number of decimal places for scores
        
    Returns:
        Dictionary mapping noise levels to formatted DataFrames
    """
    formatted = {}
    for noise, df in results_by_noise.items():
        df_copy = df.copy()
        df_copy['Mean Score'] = df_copy['Mean Score'].round(decimals)
        formatted[noise] = df_copy
    return formatted