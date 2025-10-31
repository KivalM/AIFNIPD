"""
Statistical testing utilities for comparing agent performance.

This module provides functions for:
- Variance comparison tests (Levene's test, Bartlett's test, F-test)
- Performance comparison tests (t-tests, Mann-Whitney U, ANOVA, Kruskal-Wallis)
- Multiple comparison corrections (Bonferroni, Holm-Bonferroni)
- Effect size calculations (Cohen's d, Hedges' g)
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations


def levene_test(
    *samples: np.ndarray,
    center: str = 'median'
) -> Tuple[float, float]:
    """
    Perform Levene's test for equality of variances.
    
    Levene's test is robust to departures from normality.
    
    Args:
        *samples: Variable number of sample arrays
        center: {'mean', 'median', 'trimmed'} - which function to use for center
        
    Returns:
        Tuple of (test_statistic, p_value)
    """
    statistic, p_value = stats.levene(*samples, center=center)
    return statistic, p_value


def bartlett_test(*samples: np.ndarray) -> Tuple[float, float]:
    """
    Perform Bartlett's test for equality of variances.
    
    Bartlett's test is more powerful than Levene's but sensitive to departures from normality.
    
    Args:
        *samples: Variable number of sample arrays
        
    Returns:
        Tuple of (test_statistic, p_value)
    """
    statistic, p_value = stats.bartlett(*samples)
    return statistic, p_value


def f_test_variance(sample1: np.ndarray, sample2: np.ndarray) -> Tuple[float, float]:
    """
    Perform F-test for equality of variances between two samples.
    
    Args:
        sample1: First sample array
        sample2: Second sample array
        
    Returns:
        Tuple of (f_statistic, p_value)
    """
    var1 = np.var(sample1, ddof=1)
    var2 = np.var(sample2, ddof=1)
    
    # F statistic is ratio of variances (larger / smaller)
    if var1 >= var2:
        f_stat = var1 / var2
        df1 = len(sample1) - 1
        df2 = len(sample2) - 1
    else:
        f_stat = var2 / var1
        df1 = len(sample2) - 1
        df2 = len(sample1) - 1
    
    # Two-tailed p-value
    p_value = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))
    
    return f_stat, p_value


def compare_variances_pairwise(
    data_dict: Dict[str, np.ndarray],
    test: str = 'levene',
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Perform pairwise variance comparison tests between multiple groups.
    
    Args:
        data_dict: Dictionary mapping group names to sample arrays
        test: {'levene', 'bartlett', 'f'} - which test to use
        alpha: Significance level
        
    Returns:
        DataFrame with pairwise comparison results
    """
    results = []
    group_names = list(data_dict.keys())
    
    for name1, name2 in combinations(group_names, 2):
        sample1 = data_dict[name1]
        sample2 = data_dict[name2]
        
        if test == 'levene':
            stat, p_val = levene_test(sample1, sample2)
        elif test == 'bartlett':
            stat, p_val = bartlett_test(sample1, sample2)
        elif test == 'f':
            stat, p_val = f_test_variance(sample1, sample2)
        else:
            raise ValueError(f"Unknown test: {test}")
        
        var1 = np.var(sample1, ddof=1)
        var2 = np.var(sample2, ddof=1)
        
        results.append({
            'group1': name1,
            'group2': name2,
            'var1': var1,
            'var2': var2,
            'var_ratio': var1 / var2,
            'test_statistic': stat,
            'p_value': p_val,
            'significant': p_val < alpha
        })
    
    return pd.DataFrame(results)


def compare_variances_all(
    data_dict: Dict[str, np.ndarray],
    test: str = 'levene',
    alpha: float = 0.05
) -> Dict[str, Union[float, bool]]:
    """
    Test if all groups have equal variances (omnibus test).
    
    Args:
        data_dict: Dictionary mapping group names to sample arrays
        test: {'levene', 'bartlett'} - which test to use
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    samples = list(data_dict.values())
    
    if test == 'levene':
        stat, p_val = levene_test(*samples)
    elif test == 'bartlett':
        stat, p_val = bartlett_test(*samples)
    else:
        raise ValueError(f"Unknown test: {test}. Use 'levene' or 'bartlett' for omnibus tests.")
    
    return {
        'test': test,
        'statistic': stat,
        'p_value': p_val,
        'significant': p_val < alpha,
        'n_groups': len(data_dict)
    }


def ttest_independent(
    sample1: np.ndarray,
    sample2: np.ndarray,
    equal_var: bool = True
) -> Tuple[float, float]:
    """
    Perform independent samples t-test.
    
    Args:
        sample1: First sample array
        sample2: Second sample array
        equal_var: If True, perform standard t-test. If False, perform Welch's t-test.
        
    Returns:
        Tuple of (t_statistic, p_value)
    """
    statistic, p_value = stats.ttest_ind(sample1, sample2, equal_var=equal_var)
    return statistic, p_value


def mann_whitney_u_test(
    sample1: np.ndarray,
    sample2: np.ndarray,
    alternative: str = 'two-sided'
) -> Tuple[float, float]:
    """
    Perform Mann-Whitney U test (non-parametric alternative to t-test).
    
    Args:
        sample1: First sample array
        sample2: Second sample array
        alternative: {'two-sided', 'less', 'greater'}
        
    Returns:
        Tuple of (u_statistic, p_value)
    """
    statistic, p_value = stats.mannwhitneyu(sample1, sample2, alternative=alternative)
    return statistic, p_value


def cohens_d(sample1: np.ndarray, sample2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.
    
    Args:
        sample1: First sample array
        sample2: Second sample array
        
    Returns:
        Cohen's d effect size
    """
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    std1 = np.std(sample1, ddof=1)
    std2 = np.std(sample2, ddof=1)
    n1 = len(sample1)
    n2 = len(sample2)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    d = (mean1 - mean2) / pooled_std
    return d


def hedges_g(sample1: np.ndarray, sample2: np.ndarray) -> float:
    """
    Calculate Hedges' g effect size (corrected Cohen's d for small samples).
    
    Args:
        sample1: First sample array
        sample2: Second sample array
        
    Returns:
        Hedges' g effect size
    """
    d = cohens_d(sample1, sample2)
    n1 = len(sample1)
    n2 = len(sample2)
    
    # Correction factor
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    g = d * correction
    
    return g


def compare_means_pairwise(
    data_dict: Dict[str, np.ndarray],
    test: str = 'ttest',
    equal_var: bool = True,
    alpha: float = 0.05,
    correction: Optional[str] = None
) -> pd.DataFrame:
    """
    Perform pairwise mean comparison tests between multiple groups.
    
    Args:
        data_dict: Dictionary mapping group names to sample arrays
        test: {'ttest', 'welch', 'mann_whitney'} - which test to use
        equal_var: Whether to assume equal variances (for t-test)
        alpha: Significance level
        correction: {'bonferroni', 'holm'} - multiple comparison correction method
        
    Returns:
        DataFrame with pairwise comparison results including effect sizes
    """
    results = []
    group_names = list(data_dict.keys())
    
    for name1, name2 in combinations(group_names, 2):
        sample1 = data_dict[name1]
        sample2 = data_dict[name2]
        
        if test == 'ttest':
            stat, p_val = ttest_independent(sample1, sample2, equal_var=True)
        elif test == 'welch':
            stat, p_val = ttest_independent(sample1, sample2, equal_var=False)
        elif test == 'mann_whitney':
            stat, p_val = mann_whitney_u_test(sample1, sample2)
        else:
            raise ValueError(f"Unknown test: {test}")
        
        # Calculate effect sizes
        d = cohens_d(sample1, sample2)
        g = hedges_g(sample1, sample2)
        
        mean1 = np.mean(sample1)
        mean2 = np.mean(sample2)
        
        results.append({
            'group1': name1,
            'group2': name2,
            'mean1': mean1,
            'mean2': mean2,
            'mean_diff': mean1 - mean2,
            'test_statistic': stat,
            'p_value': p_val,
            'cohens_d': d,
            'hedges_g': g,
        })
    
    df = pd.DataFrame(results)
    
    # Apply multiple comparison correction if requested
    if correction:
        df = apply_multiple_comparison_correction(df, alpha, correction)
    else:
        df['significant'] = df['p_value'] < alpha
    
    return df


def apply_multiple_comparison_correction(
    results_df: pd.DataFrame,
    alpha: float = 0.05,
    method: str = 'bonferroni'
) -> pd.DataFrame:
    """
    Apply multiple comparison correction to p-values.
    
    Args:
        results_df: DataFrame with 'p_value' column
        alpha: Significance level
        method: {'bonferroni', 'holm'} - correction method
        
    Returns:
        DataFrame with corrected p-values and significance
    """
    df = results_df.copy()
    p_values = df['p_value'].values
    n_comparisons = len(p_values)
    
    if method == 'bonferroni':
        corrected_alpha = alpha / n_comparisons
        df['corrected_alpha'] = corrected_alpha
        df['significant'] = p_values < corrected_alpha
    
    elif method == 'holm':
        # Sort by p-value
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        
        # Calculate Holm-Bonferroni adjusted alphas
        adjusted_alphas = alpha / (n_comparisons - np.arange(n_comparisons))
        
        # Determine significance
        significant = np.zeros(n_comparisons, dtype=bool)
        for i, (p_val, adj_alpha) in enumerate(zip(sorted_p_values, adjusted_alphas)):
            if p_val < adj_alpha:
                significant[sorted_indices[i]] = True
            else:
                # Once we fail to reject, all subsequent tests are not significant
                break
        
        df['significant'] = significant
    
    else:
        raise ValueError(f"Unknown correction method: {method}")
    
    df['correction_method'] = method
    return df


def anova_oneway(data_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Perform one-way ANOVA to test if means differ across groups.
    
    Args:
        data_dict: Dictionary mapping group names to sample arrays
        
    Returns:
        Dictionary with test results
    """
    samples = list(data_dict.values())
    f_stat, p_value = stats.f_oneway(*samples)
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'n_groups': len(data_dict)
    }


def kruskal_wallis_test(data_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Perform Kruskal-Wallis H test (non-parametric alternative to ANOVA).
    
    Args:
        data_dict: Dictionary mapping group names to sample arrays
        
    Returns:
        Dictionary with test results
    """
    samples = list(data_dict.values())
    h_stat, p_value = stats.kruskal(*samples)
    
    return {
        'h_statistic': h_stat,
        'p_value': p_value,
        'n_groups': len(data_dict)
    }


def load_agent_repetition_scores(
    pool_dir: Path,
    agent_dir_name: str,
    noise_level: float
) -> np.ndarray:
    """
    Load all repetition scores for a specific agent at a given noise level.
    
    Args:
        pool_dir: Path to pool directory
        agent_dir_name: Agent directory name
        noise_level: Noise level
        
    Returns:
        Array of scores (one per repetition)
    """
    from .result_utils import load_tournament_results
    
    df = load_tournament_results(pool_dir, agent_dir_name, noise_level)
    if df.empty:
        return np.array([])
    
    # Get mean score per repetition for player index 0
    scores = df[df['Player index'] == 0].groupby('repetition')['Score'].mean().values
    return scores


def compare_agents_at_noise_level(
    pool_dir: Path,
    agent_names: List[str],
    noise_level: float,
    test_variance: bool = True,
    test_means: bool = True,
    alpha: float = 0.05
) -> Dict[str, pd.DataFrame]:
    """
    Compare multiple agents at a specific noise level using various statistical tests.
    
    Args:
        pool_dir: Path to pool directory
        agent_names: List of agent directory names
        noise_level: Noise level to analyze
        test_variance: Whether to perform variance comparison tests
        test_means: Whether to perform mean comparison tests
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    # Load data for all agents
    data_dict = {}
    for agent_name in agent_names:
        scores = load_agent_repetition_scores(pool_dir, agent_name, noise_level)
        if len(scores) > 0:
            data_dict[agent_name] = scores
    
    results = {}
    
    if test_variance and len(data_dict) >= 2:
        # Omnibus variance test
        results['variance_omnibus'] = pd.DataFrame([
            compare_variances_all(data_dict, test='levene', alpha=alpha)
        ])
        
        # Pairwise variance comparisons
        results['variance_pairwise'] = compare_variances_pairwise(
            data_dict, test='levene', alpha=alpha
        )
    
    if test_means and len(data_dict) >= 2:
        # Omnibus mean test
        results['means_omnibus_anova'] = pd.DataFrame([
            anova_oneway(data_dict)
        ])
        results['means_omnibus_kruskal'] = pd.DataFrame([
            kruskal_wallis_test(data_dict)
        ])
        
        # Pairwise mean comparisons
        # First check if variances are equal to decide on test
        if test_variance:
            levene_result = compare_variances_all(data_dict, test='levene', alpha=alpha)
            equal_var = not levene_result['significant']
        else:
            equal_var = True
        
        test_type = 'ttest' if equal_var else 'welch'
        results['means_pairwise_parametric'] = compare_means_pairwise(
            data_dict, test=test_type, alpha=alpha, correction='holm'
        )
        results['means_pairwise_nonparametric'] = compare_means_pairwise(
            data_dict, test='mann_whitney', alpha=alpha, correction='holm'
        )
    
    return results


def create_variance_comparison_summary(
    pool_dir: Path,
    agent_names: List[str],
    noise_levels: List[float]
) -> pd.DataFrame:
    """
    Create a summary table of variance comparisons across noise levels.
    
    Args:
        pool_dir: Path to pool directory
        agent_names: List of agent directory names
        noise_levels: List of noise levels to analyze
        
    Returns:
        DataFrame summarizing variance tests across noise levels
    """
    results = []
    
    for noise in noise_levels:
        # Load data
        data_dict = {}
        for agent_name in agent_names:
            scores = load_agent_repetition_scores(pool_dir, agent_name, noise)
            if len(scores) > 0:
                data_dict[agent_name] = scores
        
        if len(data_dict) >= 2:
            # Omnibus test
            omnibus = compare_variances_all(data_dict, test='levene')
            
            # Calculate coefficient of variation for each agent
            for agent_name, scores in data_dict.items():
                cv = np.std(scores, ddof=1) / np.mean(scores)
                
                results.append({
                    'noise_level': noise,
                    'agent': agent_name,
                    'mean': np.mean(scores),
                    'variance': np.var(scores, ddof=1),
                    'std': np.std(scores, ddof=1),
                    'cv': cv,
                    'n': len(scores),
                    'levene_statistic': omnibus['statistic'],
                    'levene_p_value': omnibus['p_value'],
                    'variances_equal': not omnibus['significant']
                })
    
    return pd.DataFrame(results)

