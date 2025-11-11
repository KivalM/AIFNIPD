"""
Plotting utilities for creating publication-quality figures from tournament results.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def setup_publication_style(use_latex: bool = True, font_size: int = 10):
    """
    Configure matplotlib for publication-quality plots.
    
    Args:
        use_latex: Whether to use LaTeX text rendering (requires LaTeX installation)
        font_size: Base font size for plots
    """
    if use_latex:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        })
    
    plt.rcParams.update({
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 1,
        "legend.fontsize": font_size - 2,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
    })


def get_default_colors_and_markers() -> Tuple[List, List]:
    """
    Get default color palette and marker styles for consistent plotting.
    
    Returns:
        Tuple of (colors, markers) lists
    """
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    return colors, markers


def filter_data_by_agents(
    data: pd.DataFrame,
    agent_indices: Optional[List[Union[int, str]]] = None,
    rename_map: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Filter and optionally rename agents in a DataFrame.
    
    Args:
        data: DataFrame with 'agent_name' column
        agent_indices: List of agent indices (int) or name prefixes (str) to keep.
                      If None, keep all agents.
        rename_map: Optional mapping from old agent names to new display names
        
    Returns:
        Filtered and optionally renamed DataFrame
    """
    df = data.copy()
    
    # Filter by agent indices if provided
    if agent_indices is not None:
        def keep_agent(agent_name):
            # Extract the index from agent name (format: "index_AgentName")
            try:
                agent_index = int(agent_name.split('_')[0])
                return agent_index in agent_indices
            except (ValueError, IndexError):
                return False
        df = df[df['agent_name'].apply(keep_agent)]
    
    # Rename agents if mapping provided
    if rename_map is not None:
        def rename_agent(agent_name):
            # Extract the index from agent name (format: "index_AgentName")
            try:
                agent_index = agent_name.split('_')[0]
                if agent_index in rename_map:
                    return rename_map[agent_index]
            except (ValueError, IndexError):
                pass
            return agent_name
        df['agent_name'] = df['agent_name'].apply(rename_agent)
    
    return df


def plot_metric_vs_noise(
    data: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    figsize: Tuple[float, float] = (3.5, 2.8),
    ylim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    xlabel: str = r'Noise Level ($\epsilon$)',
    show_legend: bool = True,
    legend_kwargs: Optional[Dict] = None,
    save_path: Optional[Path] = None,
    save_formats: List[str] = ['pdf', 'png']
) -> Tuple[Figure, Axes]:
    """
    Create a publication-quality plot of a metric vs noise level with confidence intervals.
    
    Args:
        data: DataFrame with columns: agent_name, noise_level, <metric_col>, ci_lower, ci_upper
        metric_col: Name of the metric column to plot (e.g., 'mean_score', 'mean_cc_rate')
        ylabel: Y-axis label
        title: Plot title
        figsize: Figure size in inches (width, height)
        ylim: Y-axis limits (min, max)
        xlim: X-axis limits (min, max)
        xlabel: X-axis label
        show_legend: Whether to show legend
        legend_kwargs: Additional kwargs for legend
        save_path: If provided, save figure to this path (without extension)
        save_formats: List of formats to save (default: ['pdf'])
        
    Returns:
        Tuple of (figure, axes) for further customization
    """
    colors, markers = get_default_colors_and_markers()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    agent_list = sorted(data['agent_name'].unique())
    
    for idx, agent in enumerate(agent_list):
        agent_data = data[data['agent_name'] == agent].sort_values('noise_level')
        
        # Calculate asymmetric error bars
        yerr_lower = agent_data[metric_col] - agent_data['ci_lower']
        yerr_upper = agent_data['ci_upper'] - agent_data[metric_col]
        yerr = [yerr_lower.values, yerr_upper.values]
        
        ax.errorbar(
            agent_data['noise_level'], 
            agent_data[metric_col],
            yerr=yerr, 
            marker=markers[idx % len(markers)],
            label=agent,
            linewidth=1.2,
            markersize=3.5,
            capsize=2,
            capthick=0.8,
            elinewidth=0.8,
            alpha=0.85,
            color=colors[idx % len(colors)]
        )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    if show_legend:
        legend_defaults = {'loc': 'best', 'frameon': True, 'framealpha': 0.9, 'edgecolor': 'gray'}
        if legend_kwargs:
            legend_defaults.update(legend_kwargs)
        ax.legend(**legend_defaults)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        for fmt in save_formats:
            save_file = Path(str(save_path) + f'.{fmt}')
            plt.savefig(save_file, format=fmt, dpi=600, bbox_inches='tight')
    
    return fig, ax


def plot_scores_vs_noise(
    scores_data: pd.DataFrame,
    title: str = 'Agent Score vs Noise Level',
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot agent scores vs noise level with confidence intervals.
    
    Args:
        scores_data: DataFrame with columns: agent_name, noise_level, mean_score, ci_lower, ci_upper
        title: Plot title
        **kwargs: Additional arguments passed to plot_metric_vs_noise
        
    Returns:
        Tuple of (figure, axes)
    """
    return plot_metric_vs_noise(
        data=scores_data,
        metric_col='mean_score',
        ylabel='Mean Score',
        title=title,
        xlim=kwargs.pop('xlim', (-0.015, 0.265)),
        **kwargs
    )


def plot_cc_rate_vs_noise(
    cc_data: pd.DataFrame,
    title: str = 'CC Rate vs Noise Level',
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot cooperation-cooperation rate vs noise level with confidence intervals.
    
    Args:
        cc_data: DataFrame with columns: agent_name, noise_level, mean_cc_rate, ci_lower, ci_upper
        title: Plot title
        **kwargs: Additional arguments passed to plot_metric_vs_noise
        
    Returns:
        Tuple of (figure, axes)
    """
    return plot_metric_vs_noise(
        data=cc_data,
        metric_col='mean_cc_rate',
        ylabel='CC Rate',
        title=title,
        ylim=kwargs.pop('ylim', (-0.05, 1.05)),
        xlim=kwargs.pop('xlim', (-0.015, 0.265)),
        **kwargs
    )


def plot_cd_rate_vs_noise(
    cd_data: pd.DataFrame,
    title: str = 'CD Rate vs Noise Level',
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot cooperation-defection rate vs noise level with confidence intervals.
    
    Args:
        cd_data: DataFrame with columns: agent_name, noise_level, mean_cd_rate, ci_lower, ci_upper
        title: Plot title
        **kwargs: Additional arguments passed to plot_metric_vs_noise
        
    Returns:
        Tuple of (figure, axes)
    """
    return plot_metric_vs_noise(
        data=cd_data,
        metric_col='mean_cd_rate',
        ylabel='CD Rate',
        title=title,
        ylim=kwargs.pop('ylim', (-0.05, 0.45)),
        xlim=kwargs.pop('xlim', (-0.015, 0.265)),
        **kwargs
    )


def plot_normalized_cooperation_vs_noise(
    norm_coop_data: pd.DataFrame,
    title: str = 'Normalized Cooperation vs Noise Level',
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot normalized cooperation rate (CC / (CC + CD)) vs noise level with confidence intervals.
    
    Args:
        norm_coop_data: DataFrame with columns: agent_name, noise_level, mean_norm_coop, ci_lower, ci_upper
        title: Plot title
        **kwargs: Additional arguments passed to plot_metric_vs_noise
        
    Returns:
        Tuple of (figure, axes)
    """
    return plot_metric_vs_noise(
        data=norm_coop_data,
        metric_col='mean_norm_coop',
        ylabel='Normalized Cooperation',
        title=title,
        ylim=kwargs.pop('ylim', (-0.05, 1.05)),
        xlim=kwargs.pop('xlim', (-0.015, 0.265)),
        **kwargs
    )


def create_comparison_plots(
    static_data: pd.DataFrame,
    learning_data: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    figsize: Tuple[float, float] = (7.2, 2.8),
    ylim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    static_title: str = 'Static Pool',
    learning_title: str = 'Learning Pool',
    save_path: Optional[Path] = None,
    save_formats: List[str] = ['pdf']
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """
    Create side-by-side comparison plots for static and learning pools.
    
    Args:
        static_data: DataFrame for static pool
        learning_data: DataFrame for learning pool
        metric_col: Metric column to plot
        ylabel: Y-axis label
        figsize: Figure size (width, height)
        ylim: Y-axis limits
        xlim: X-axis limits
        static_title: Title for static pool subplot
        learning_title: Title for learning pool subplot
        save_path: Path to save figure (without extension)
        save_formats: Formats to save
        
    Returns:
        Tuple of (figure, (ax1, ax2))
    """
    colors, markers = get_default_colors_and_markers()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Static pool
    for idx, agent in enumerate(sorted(static_data['agent_name'].unique())):
        agent_data = static_data[static_data['agent_name'] == agent].sort_values('noise_level')
        yerr_lower = agent_data[metric_col] - agent_data['ci_lower']
        yerr_upper = agent_data['ci_upper'] - agent_data[metric_col]
        yerr = [yerr_lower.values, yerr_upper.values]
        
        ax1.errorbar(
            agent_data['noise_level'], agent_data[metric_col],
            yerr=yerr, marker=markers[idx % len(markers)], label=agent,
            linewidth=1.2, markersize=3.5, capsize=2, capthick=0.8,
            elinewidth=0.8, alpha=0.85, color=colors[idx % len(colors)]
        )
    
    ax1.set_xlabel(r'Noise Level ($\epsilon$)')
    ax1.set_ylabel(ylabel)
    ax1.set_title(static_title, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.3)
    if xlim:
        ax1.set_xlim(xlim)
    if ylim:
        ax1.set_ylim(ylim)
    
    # Learning pool
    for idx, agent in enumerate(sorted(learning_data['agent_name'].unique())):
        agent_data = learning_data[learning_data['agent_name'] == agent].sort_values('noise_level')
        yerr_lower = agent_data[metric_col] - agent_data['ci_lower']
        yerr_upper = agent_data['ci_upper'] - agent_data[metric_col]
        yerr = [yerr_lower.values, yerr_upper.values]
        
        ax2.errorbar(
            agent_data['noise_level'], agent_data[metric_col],
            yerr=yerr, marker=markers[idx % len(markers)], label=agent,
            linewidth=1.2, markersize=3.5, capsize=2, capthick=0.8,
            elinewidth=0.8, alpha=0.85, color=colors[idx % len(colors)]
        )
    
    ax2.set_xlabel(r'Noise Level ($\epsilon$)')
    ax2.set_ylabel(ylabel)
    ax2.set_title(learning_title, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.3)
    if xlim:
        ax2.set_xlim(xlim)
    if ylim:
        ax2.set_ylim(ylim)
    
    plt.tight_layout()
    
    if save_path:
        for fmt in save_formats:
            save_file = Path(str(save_path) + f'.{fmt}')
            plt.savefig(save_file, format=fmt, dpi=600, bbox_inches='tight')
    
    return fig, (ax1, ax2)


def generate_plots_for_agent_groups(
    data: pd.DataFrame,
    agent_groups: List[List[Union[int, str]]],
    plot_function: callable,
    rename_maps: Optional[List[Dict[str, str]]] = None,
    base_save_path: Optional[Path] = None,
    **plot_kwargs
) -> List[Tuple[Figure, Axes]]:
    """
    Generate multiple plots, one for each group of agents.
    
    This is useful when you have many agents and want to create separate plots
    for different subsets to avoid overcrowding.
    
    Args:
        data: DataFrame with agent data
        agent_groups: List of lists, where each inner list contains agent indices/prefixes to plot together
        plot_function: Function to use for plotting (e.g., plot_scores_vs_noise)
        rename_maps: Optional list of rename maps, one per group
        base_save_path: Base path for saving (will append _group0, _group1, etc.)
        **plot_kwargs: Additional arguments passed to plot_function
        
    Returns:
        List of (figure, axes) tuples, one per group
        
    Example:
        >>> agent_groups = [[0, 1, 2], [3, 4, 5], [6, 7]]
        >>> figs = generate_plots_for_agent_groups(
        ...     scores_data, 
        ...     agent_groups,
        ...     plot_scores_vs_noise,
        ...     base_save_path=Path('results/score_comparison')
        ... )
    """
    results = []
    
    for i, agent_group in enumerate(agent_groups):
        # Filter data for this group
        rename_map = rename_maps[i] if rename_maps and i < len(rename_maps) else None
        filtered_data = filter_data_by_agents(data, agent_group, rename_map)
        
        # Determine save path for this group
        if base_save_path:
            group_save_path = Path(str(base_save_path) + f'_group{i}')
        else:
            group_save_path = None
        
        # Create plot
        fig, ax = plot_function(
            filtered_data,
            save_path=group_save_path,
            **plot_kwargs
        )
        
        results.append((fig, ax))
    
    return results


def create_summary_table(
    scores_data: pd.DataFrame,
    cc_data: pd.DataFrame,
    cd_data: pd.DataFrame,
    pool_name: str = 'Static'
) -> pd.DataFrame:
    """
    Create a comprehensive summary table for all agents across all metrics and noise levels.
    
    Args:
        scores_data: DataFrame with score data
        cc_data: DataFrame with CC rate data
        cd_data: DataFrame with CD rate data
        pool_name: Name of the pool (e.g., 'Static', 'Learning')
        
    Returns:
        DataFrame with comprehensive metrics for each agent and noise level
    """
    summary_data = []
    
    for agent in scores_data['agent_name'].unique():
        for noise in scores_data['noise_level'].unique():
            # Get data for this agent and noise level
            score_row = scores_data[
                (scores_data['agent_name'] == agent) & 
                (scores_data['noise_level'] == noise)
            ]
            cc_row = cc_data[
                (cc_data['agent_name'] == agent) & 
                (cc_data['noise_level'] == noise)
            ]
            cd_row = cd_data[
                (cd_data['agent_name'] == agent) & 
                (cd_data['noise_level'] == noise)
            ]
            
            if not score_row.empty and not cc_row.empty and not cd_row.empty:
                summary_data.append({
                    'pool': pool_name,
                    'agent': agent,
                    'noise_level': noise,
                    'score_mean': score_row['mean_score'].values[0],
                    'score_std': score_row['std_score'].values[0],
                    'score_ci_lower': score_row['ci_lower'].values[0],
                    'score_ci_upper': score_row['ci_upper'].values[0],
                    'cc_rate_mean': cc_row['mean_cc_rate'].values[0],
                    'cc_rate_std': cc_row['std_cc_rate'].values[0],
                    'cc_rate_ci_lower': cc_row['ci_lower'].values[0],
                    'cc_rate_ci_upper': cc_row['ci_upper'].values[0],
                    'cd_rate_mean': cd_row['mean_cd_rate'].values[0],
                    'cd_rate_std': cd_row['std_cd_rate'].values[0],
                    'cd_rate_ci_lower': cd_row['ci_lower'].values[0],
                    'cd_rate_ci_upper': cd_row['ci_upper'].values[0],
                })
    
    return pd.DataFrame(summary_data).sort_values(['pool', 'agent', 'noise_level'])


def create_formatted_table(
    summary_table: pd.DataFrame,
    score_decimals: int = 2,
    rate_decimals: int = 3
) -> pd.DataFrame:
    """
    Create a human-readable formatted table with confidence intervals.
    
    Args:
        summary_table: Output from create_summary_table
        score_decimals: Number of decimals for score values
        rate_decimals: Number of decimals for rate values
        
    Returns:
        DataFrame with formatted strings for easy reading
    """
    formatted_data = []
    
    for _, row in summary_table.iterrows():
        formatted_data.append({
            'Pool': row['pool'],
            'Agent': row['agent'],
            'Noise': f"{row['noise_level']:.2f}",
            'Score': f"{row['score_mean']:.{score_decimals}f} ± {row['score_std']:.{score_decimals}f}",
            'Score_CI': f"[{row['score_ci_lower']:.{score_decimals}f}, {row['score_ci_upper']:.{score_decimals}f}]",
            'CC_Rate': f"{row['cc_rate_mean']:.{rate_decimals}f} ± {row['cc_rate_std']:.{rate_decimals}f}",
            'CC_Rate_CI': f"[{row['cc_rate_ci_lower']:.{rate_decimals}f}, {row['cc_rate_ci_upper']:.{rate_decimals}f}]",
            'CD_Rate': f"{row['cd_rate_mean']:.{rate_decimals}f} ± {row['cd_rate_std']:.{rate_decimals}f}",
            'CD_Rate_CI': f"[{row['cd_rate_ci_lower']:.{rate_decimals}f}, {row['cd_rate_ci_upper']:.{rate_decimals}f}]",
        })
    
    return pd.DataFrame(formatted_data)


def create_ranking_table(
    scores_data: pd.DataFrame,
    pool_name: str = 'Static'
) -> pd.DataFrame:
    """
    Create a ranking table showing agent ranks at each noise level.
    
    Args:
        scores_data: DataFrame with score data
        pool_name: Name of the pool
        
    Returns:
        DataFrame with rankings for each noise level
    """
    rankings_data = []
    
    for noise in sorted(scores_data['noise_level'].unique()):
        noise_data = scores_data[scores_data['noise_level'] == noise].copy()
        noise_data = noise_data.sort_values('mean_score', ascending=False).reset_index(drop=True)
        noise_data['rank'] = range(1, len(noise_data) + 1)
        
        for _, row in noise_data.iterrows():
            rankings_data.append({
                'pool': pool_name,
                'noise_level': noise,
                'rank': row['rank'],
                'agent': row['agent_name'],
                'mean_score': row['mean_score'],
                'ci_lower': row['ci_lower'],
                'ci_upper': row['ci_upper']
            })
    
    return pd.DataFrame(rankings_data)

