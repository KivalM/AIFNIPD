"""
Utility functions for processing hyperparameter experiment results.
"""
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats


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


def get_agent_dirs(pool_dir: Path) -> List[Tuple[str, str]]:
    """
    Get all agent directories from a pool directory.
    
    Args:
        pool_dir: Path to the pool directory (e.g., results/main/static)
        
    Returns:
        List of (directory_name, display_name) tuples sorted by index
        display_name includes the index to distinguish agents with same class name
    """
    agent_dirs = []
    for d in pool_dir.iterdir():
        if d.is_dir():
            # Format is "index_AgentName"
            # Use the full directory name as display name to keep index
            agent_dirs.append((d.name, d.name))
    
    # Sort by index
    agent_dirs.sort(key=lambda x: int(x[0].split('_')[0]))
    return agent_dirs


def load_tournament_results(
    pool_dir: Path,
    agent_dir_name: str,
    noise_level: float
) -> pd.DataFrame:
    """
    Load tournament results for a specific agent and noise level across all repetitions.
    
    Args:
        pool_dir: Path to the pool directory (e.g., results/main/static)
        agent_dir_name: Agent directory name (e.g., "0_DBS")
        noise_level: Noise level to load
        
    Returns:
        DataFrame with all repetitions combined
    """
    agent_path = pool_dir / agent_dir_name
    all_dfs = []
    
    for rep_dir in sorted(agent_path.glob('repetition_*')):
        csv_file = rep_dir / f"{noise_level}.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            df['repetition'] = int(rep_dir.name.split('_')[1])
            all_dfs.append(df)
    
    if not all_dfs:
        return pd.DataFrame()
    
    return pd.concat(all_dfs, ignore_index=True)


def calculate_agent_score(
    pool_dir: Path,
    agent_dir_name: str,
    noise_levels: List[float]
) -> pd.DataFrame:
    """
    Calculate mean score for an agent across all noise levels and repetitions.
    
    Args:
        pool_dir: Path to the pool directory
        agent_dir_name: Agent directory name
        noise_levels: List of noise levels to analyze
        
    Returns:
        DataFrame with columns: noise_level, mean_score, std_score, ci_lower, ci_upper, num_repetitions
    """
    results = []
    
    for noise in noise_levels:
        df = load_tournament_results(pool_dir, agent_dir_name, noise)
        if df.empty:
            continue
        
        # Get scores for player index 0 (the agent being tested)
        agent_scores = df[df['Player index'] == 0].groupby('repetition')['Score'].mean()
        
        mean_score = agent_scores.mean()
        std_score = agent_scores.std()
        n = len(agent_scores)
        
        # Calculate 95% confidence interval using t-distribution
        if n > 1:
            ci = stats.t.interval(0.95, n-1, loc=mean_score, scale=std_score/np.sqrt(n))
            ci_lower = ci[0]
            ci_upper = ci[1]
        else:
            ci_lower = mean_score
            ci_upper = mean_score
        
        results.append({
            'noise_level': noise,
            'mean_score': mean_score,
            'std_score': std_score,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'num_repetitions': n
        })
    
    return pd.DataFrame(results)


def calculate_cc_rate(
    pool_dir: Path,
    agent_dir_name: str,
    noise_levels: List[float]
) -> pd.DataFrame:
    """
    Calculate cooperation-cooperation rate for an agent across all noise levels and repetitions.
    
    Args:
        pool_dir: Path to the pool directory
        agent_dir_name: Agent directory name
        noise_levels: List of noise levels to analyze
        
    Returns:
        DataFrame with columns: noise_level, mean_cc_rate, std_cc_rate, ci_lower, ci_upper, num_repetitions
    """
    results = []
    
    for noise in noise_levels:
        df = load_tournament_results(pool_dir, agent_dir_name, noise)
        if df.empty:
            continue
        
        # Get CC rate for player index 0
        agent_data = df[df['Player index'] == 0]
        
        # Calculate CC rate per repetition
        cc_rates = []
        for rep in agent_data['repetition'].unique():
            rep_data = agent_data[agent_data['repetition'] == rep]
            total_turns = rep_data['Turns'].sum()
            cc_count = rep_data['CC count'].sum()
            if total_turns > 0:
                cc_rates.append(cc_count / total_turns)
        
        mean_cc_rate = np.mean(cc_rates) if cc_rates else 0
        std_cc_rate = np.std(cc_rates, ddof=1) if len(cc_rates) > 1 else 0
        n = len(cc_rates)
        
        # Calculate 95% confidence interval using t-distribution
        if n > 1:
            ci = stats.t.interval(0.95, n-1, loc=mean_cc_rate, scale=std_cc_rate/np.sqrt(n))
            ci_lower = max(0, ci[0])  # CC rate can't be negative
            ci_upper = min(1, ci[1])  # CC rate can't exceed 1
        else:
            ci_lower = mean_cc_rate
            ci_upper = mean_cc_rate
        
        results.append({
            'noise_level': noise,
            'mean_cc_rate': mean_cc_rate,
            'std_cc_rate': std_cc_rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'num_repetitions': n
        })
    
    return pd.DataFrame(results)


def load_all_agents_scores(
    pool_dir: Path,
    noise_levels: List[float]
) -> pd.DataFrame:
    """
    Load scores for all agents in a pool.
    
    Args:
        pool_dir: Path to the pool directory
        noise_levels: List of noise levels to analyze
        
    Returns:
        DataFrame with columns: agent_name, noise_level, mean_score, std_score, ci_lower, ci_upper
        agent_name includes index (e.g., "0_DBS", "1_DBS") to distinguish duplicates
    """
    agent_dirs = get_agent_dirs(pool_dir)
    all_results = []
    
    for dir_name, display_name in agent_dirs:
        scores_df = calculate_agent_score(pool_dir, dir_name, noise_levels)
        scores_df['agent_name'] = display_name
        all_results.append(scores_df)
    
    if not all_results:
        return pd.DataFrame()
    
    result_df = pd.concat(all_results, ignore_index=True)
    return result_df[['agent_name', 'noise_level', 'mean_score', 'std_score', 'ci_lower', 'ci_upper', 'num_repetitions']]


def load_all_agents_cc_rates(
    pool_dir: Path,
    noise_levels: List[float]
) -> pd.DataFrame:
    """
    Load CC rates for all agents in a pool.
    
    Args:
        pool_dir: Path to the pool directory
        noise_levels: List of noise levels to analyze
        
    Returns:
        DataFrame with columns: agent_name, noise_level, mean_cc_rate, std_cc_rate, ci_lower, ci_upper
        agent_name includes index (e.g., "0_DBS", "1_DBS") to distinguish duplicates
    """
    agent_dirs = get_agent_dirs(pool_dir)
    all_results = []
    
    for dir_name, display_name in agent_dirs:
        cc_df = calculate_cc_rate(pool_dir, dir_name, noise_levels)
        cc_df['agent_name'] = display_name
        all_results.append(cc_df)
    
    if not all_results:
        return pd.DataFrame()
    
    result_df = pd.concat(all_results, ignore_index=True)
    return result_df[['agent_name', 'noise_level', 'mean_cc_rate', 'std_cc_rate', 'ci_lower', 'ci_upper', 'num_repetitions']]


def calculate_cd_rate(
    pool_dir: Path,
    agent_dir_name: str,
    noise_levels: List[float]
) -> pd.DataFrame:
    """
    Calculate cooperation-defection rate for an agent across all noise levels and repetitions.
    
    Args:
        pool_dir: Path to the pool directory
        agent_dir_name: Agent directory name
        noise_levels: List of noise levels to analyze
        
    Returns:
        DataFrame with columns: noise_level, mean_cd_rate, std_cd_rate, ci_lower, ci_upper, num_repetitions
    """
    results = []
    
    for noise in noise_levels:
        df = load_tournament_results(pool_dir, agent_dir_name, noise)
        if df.empty:
            continue
        
        # Get CD rate for player index 0
        agent_data = df[df['Player index'] == 0]
        
        # Calculate CD rate per repetition
        cd_rates = []
        for rep in agent_data['repetition'].unique():
            rep_data = agent_data[agent_data['repetition'] == rep]
            total_turns = rep_data['Turns'].sum()
            cd_count = rep_data['CD count'].sum()
            if total_turns > 0:
                cd_rates.append(cd_count / total_turns)
        
        mean_cd_rate = np.mean(cd_rates) if cd_rates else 0
        std_cd_rate = np.std(cd_rates, ddof=1) if len(cd_rates) > 1 else 0
        n = len(cd_rates)
        
        # Calculate 95% confidence interval using t-distribution
        if n > 1:
            ci = stats.t.interval(0.95, n-1, loc=mean_cd_rate, scale=std_cd_rate/np.sqrt(n))
            ci_lower = max(0, ci[0])  # CD rate can't be negative
            ci_upper = min(1, ci[1])  # CD rate can't exceed 1
        else:
            ci_lower = mean_cd_rate
            ci_upper = mean_cd_rate
        
        results.append({
            'noise_level': noise,
            'mean_cd_rate': mean_cd_rate,
            'std_cd_rate': std_cd_rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'num_repetitions': n
        })
    
    return pd.DataFrame(results)


def load_all_agents_cd_rates(
    pool_dir: Path,
    noise_levels: List[float]
) -> pd.DataFrame:
    """
    Load CD rates for all agents in a pool.
    
    Args:
        pool_dir: Path to the pool directory
        noise_levels: List of noise levels to analyze
        
    Returns:
        DataFrame with columns: agent_name, noise_level, mean_cd_rate, std_cd_rate, ci_lower, ci_upper
        agent_name includes index (e.g., "0_DBS", "1_DBS") to distinguish duplicates
    """
    agent_dirs = get_agent_dirs(pool_dir)
    all_results = []
    
    for dir_name, display_name in agent_dirs:
        cd_df = calculate_cd_rate(pool_dir, dir_name, noise_levels)
        cd_df['agent_name'] = display_name
        all_results.append(cd_df)
    
    if not all_results:
        return pd.DataFrame()
    
    result_df = pd.concat(all_results, ignore_index=True)
    return result_df[['agent_name', 'noise_level', 'mean_cd_rate', 'std_cd_rate', 'ci_lower', 'ci_upper', 'num_repetitions']]