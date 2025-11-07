"""
Utility functions for processing and visualizing self-play tournament results.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def load_selfplay_results(
    sp_dir: Path,
    noise_level: float
) -> pd.DataFrame:
    """
    Load self-play tournament results for a specific noise level across all repetitions.
    
    Args:
        sp_dir: Path to the self-play directory (e.g., results/sp)
        noise_level: Noise level to load
        
    Returns:
        DataFrame with all repetitions combined
    """
    all_dfs = []
    
    for rep_dir in sorted(sp_dir.glob('repetition_*')):
        csv_file = rep_dir / f"{noise_level}.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            df['repetition'] = int(rep_dir.name.split('_')[1])
            all_dfs.append(df)
    
    if not all_dfs:
        return pd.DataFrame()
    
    return pd.concat(all_dfs, ignore_index=True)


def get_player_names(sp_dir: Path) -> List[str]:
    """
    Get the list of player names from the self-play results.
    
    Args:
        sp_dir: Path to the self-play directory
        
    Returns:
        List of player names in order
    """
    # Load the first repetition's details to get player names
    for rep_dir in sorted(sp_dir.glob('repetition_*')):
        details_file = rep_dir / 'details.json'
        if details_file.exists():
            import json
            with open(details_file, 'r') as f:
                details = json.load(f)
                return details['player_names']
    return []


def calculate_pairwise_metrics(
    df: pd.DataFrame,
    player_names: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate pairwise CC rate, CD rate, and score matrices from tournament results.
    
    Args:
        df: DataFrame with tournament results
        player_names: List of player names in order
        
    Returns:
        Tuple of (cc_rate_matrix, cd_rate_matrix, score_matrix)
        Each matrix is N×N where N is the number of players
    """
    n_players = len(player_names)
    cc_rate_matrix = np.zeros((n_players, n_players))
    cd_rate_matrix = np.zeros((n_players, n_players))
    score_matrix = np.zeros((n_players, n_players))
    
    # Create name to index mapping
    name_to_idx = {name: i for i, name in enumerate(player_names)}
    
    # Group by player and opponent
    for player_name in player_names:
        for opponent_name in player_names:
            # Filter data for this specific matchup
            matchup_df = df[
                (df['Player name'] == player_name) & 
                (df['Opponent name'] == opponent_name)
            ]
            
            if matchup_df.empty:
                continue
            
            player_idx = name_to_idx[player_name]
            opponent_idx = name_to_idx[opponent_name]
            
            # Calculate metrics
            total_turns = matchup_df['Turns'].sum()
            if total_turns > 0:
                cc_count = matchup_df['CC count'].sum()
                cd_count = matchup_df['CD count'].sum()
                
                cc_rate_matrix[player_idx, opponent_idx] = cc_count / total_turns
                cd_rate_matrix[player_idx, opponent_idx] = cd_count / total_turns
            
            # Mean score
            score_matrix[player_idx, opponent_idx] = matchup_df['Score'].mean()
    
    return cc_rate_matrix, cd_rate_matrix, score_matrix


def create_agent_labels(
    player_names: List[str],
    rename_map: Optional[Dict[int, str]] = None
) -> List[str]:
    """
    Create clean labels for agents based on their names.
    
    Args:
        player_names: List of player names from tournament
        rename_map: Optional mapping from agent index to display name
        
    Returns:
        List of clean agent labels
    """
    if rename_map is None:
        return [str(i) for i in range(len(player_names))]
    
    labels = []
    for i, name in enumerate(player_names):
        if i in rename_map:
            labels.append(rename_map[i])
        else:
            # Try to extract a clean name from the full name
            labels.append(str(i))
    
    return labels


def plot_heatmap(
    matrix: np.ndarray,
    labels: List[str],
    title: str,
    cmap: str = 'viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cbar_label: str = '',
    figsize: Tuple[int, int] = (10, 8),
    annot: bool = True,
    fmt: str = '.2f',
    save_path: Optional[Path] = None
) -> Figure:
    """
    Create a heatmap visualization of a matrix.
    
    Args:
        matrix: N×N matrix to visualize
        labels: List of labels for rows/columns
        title: Title for the plot
        cmap: Colormap to use
        vmin: Minimum value for colormap (optional)
        vmax: Maximum value for colormap (optional)
        cbar_label: Label for colorbar
        figsize: Figure size
        annot: Whether to annotate cells with values
        fmt: Format string for annotations
        save_path: Optional path to save the figure
        
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        fmt=fmt,
        cbar_kws={'label': cbar_label},
        square=True,
        linewidths=0.5,
        linecolor='white',
        ax=ax
    )
    
    ax.set_xlabel('Opponent', fontsize=12, fontweight='bold')
    ax.set_ylabel('Player', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    
    return fig


def generate_selfplay_heatmaps(
    sp_dir: Path,
    noise_levels: List[float],
    output_dir: Path,
    rename_map: Optional[Dict[int, str]] = None,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Generate heatmaps for CC rate, CD rate, and scores across all noise levels.
    
    Args:
        sp_dir: Path to self-play results directory
        noise_levels: List of noise levels to process
        output_dir: Directory to save output plots
        rename_map: Optional mapping from agent index to display name
        figsize: Figure size for each heatmap
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get player names
    player_names = get_player_names(sp_dir)
    n_players = len(player_names)
    
    if n_players == 0:
        print("No player names found in self-play results!")
        return
    
    print(f"Found {n_players} players in self-play tournament")
    
    # Create labels
    labels = create_agent_labels(player_names, rename_map)
    
    for noise in noise_levels:
        print(f"\nProcessing noise level {noise}...")
        
        # Load data
        df = load_selfplay_results(sp_dir, noise)
        if df.empty:
            print(f"  No data found for noise level {noise}")
            continue
        
        # Calculate metrics
        cc_matrix, cd_matrix, score_matrix = calculate_pairwise_metrics(df, player_names)
        
        # Create heatmaps
        # CC Rate heatmap
        print(f"  Creating CC rate heatmap...")
        plot_heatmap(
            cc_matrix,
            labels,
            f'Cooperation (CC) Rate at Noise {noise:.2f}',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            cbar_label='CC Rate',
            figsize=figsize,
            annot=True,
            fmt='.3f',
            save_path=output_dir / f'cc_rate_noise_{noise:.2f}.pdf'
        )
        plt.close()
        
        # CD Rate heatmap
        print(f"  Creating CD rate heatmap...")
        plot_heatmap(
            cd_matrix,
            labels,
            f'Defection on Cooperation (CD) Rate at Noise {noise:.2f}',
            cmap='RdYlGn_r',
            vmin=0,
            vmax=1,
            cbar_label='CD Rate',
            figsize=figsize,
            annot=True,
            fmt='.3f',
            save_path=output_dir / f'cd_rate_noise_{noise:.2f}.pdf'
        )
        plt.close()
        
        # Score heatmap
        print(f"  Creating score heatmap...")
        plot_heatmap(
            score_matrix,
            labels,
            f'Mean Score at Noise {noise:.2f}',
            cmap='viridis',
            vmin=None,
            vmax=None,
            cbar_label='Mean Score',
            figsize=figsize,
            annot=True,
            fmt='.1f',
            save_path=output_dir / f'score_noise_{noise:.2f}.pdf'
        )
        plt.close()
        
        print(f"  ✓ Saved heatmaps for noise {noise}")
    
    print(f"\n✓ All heatmaps saved to {output_dir}")


def create_combined_heatmap_grid(
    sp_dir: Path,
    noise_levels: List[float],
    output_dir: Path,
    metric: str = 'cc_rate',
    rename_map: Optional[Dict[int, str]] = None,
    max_cols: int = 3
):
    """
    Create a grid of heatmaps showing one metric across multiple noise levels.
    
    Args:
        sp_dir: Path to self-play results directory
        noise_levels: List of noise levels to process
        output_dir: Directory to save output plots
        metric: Which metric to plot ('cc_rate', 'cd_rate', or 'score')
        rename_map: Optional mapping from agent index to display name
        max_cols: Maximum number of columns in the grid
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get player names
    player_names = get_player_names(sp_dir)
    n_players = len(player_names)
    
    if n_players == 0:
        print("No player names found in self-play results!")
        return
    
    # Create labels
    labels = create_agent_labels(player_names, rename_map)
    
    # Calculate grid dimensions
    n_noise = len(noise_levels)
    n_cols = min(max_cols, n_noise)
    n_rows = int(np.ceil(n_noise / n_cols))
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4.5*n_rows))
    if n_noise == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Configure based on metric
    if metric == 'cc_rate':
        cmap = 'RdYlGn'
        vmin, vmax = 0, 1
        cbar_label = 'CC Rate'
        title_prefix = 'CC Rate'
        fmt = '.2f'
    elif metric == 'cd_rate':
        cmap = 'RdYlGn_r'
        vmin, vmax = 0, 1
        cbar_label = 'CD Rate'
        title_prefix = 'CD Rate'
        fmt = '.2f'
    else:  # score
        cmap = 'viridis'
        vmin, vmax = None, None
        cbar_label = 'Score'
        title_prefix = 'Score'
        fmt = '.1f'
    
    for idx, noise in enumerate(noise_levels):
        ax = axes[idx]
        
        # Load data
        df = load_selfplay_results(sp_dir, noise)
        if df.empty:
            ax.text(0.5, 0.5, f'No data\nfor noise {noise:.2f}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # Calculate metrics
        cc_matrix, cd_matrix, score_matrix = calculate_pairwise_metrics(df, player_names)
        
        # Select the right matrix
        if metric == 'cc_rate':
            matrix = cc_matrix
        elif metric == 'cd_rate':
            matrix = cd_matrix
        else:
            matrix = score_matrix
        
        # Create heatmap
        sns.heatmap(
            matrix,
            xticklabels=labels,
            yticklabels=labels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=fmt,
            cbar=True,
            square=True,
            linewidths=0.5,
            linecolor='white',
            ax=ax,
            cbar_kws={'label': cbar_label}
        )
        
        ax.set_title(f'Noise {noise:.2f}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Opponent', fontsize=10)
        ax.set_ylabel('Player', fontsize=10)
        
        # Rotate labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_noise, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'{title_prefix} Across Noise Levels', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / f'{metric}_grid.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved combined grid to {save_path}")


def export_matrices_to_csv(
    sp_dir: Path,
    noise_levels: List[float],
    output_dir: Path,
    rename_map: Optional[Dict[int, str]] = None
):
    """
    Export pairwise matrices to CSV files for further analysis.
    
    Args:
        sp_dir: Path to self-play results directory
        noise_levels: List of noise levels to process
        output_dir: Directory to save output CSV files
        rename_map: Optional mapping from agent index to display name
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get player names
    player_names = get_player_names(sp_dir)
    if not player_names:
        print("No player names found!")
        return
    
    labels = create_agent_labels(player_names, rename_map)
    
    for noise in noise_levels:
        print(f"Exporting matrices for noise {noise}...")
        
        df = load_selfplay_results(sp_dir, noise)
        if df.empty:
            continue
        
        cc_matrix, cd_matrix, score_matrix = calculate_pairwise_metrics(df, player_names)
        
        # Convert to DataFrames with labels
        cc_df = pd.DataFrame(cc_matrix, index=labels, columns=labels)
        cd_df = pd.DataFrame(cd_matrix, index=labels, columns=labels)
        score_df = pd.DataFrame(score_matrix, index=labels, columns=labels)
        
        # Save to CSV
        cc_df.to_csv(output_dir / f'cc_rate_noise_{noise:.2f}.csv')
        cd_df.to_csv(output_dir / f'cd_rate_noise_{noise:.2f}.csv')
        score_df.to_csv(output_dir / f'score_noise_{noise:.2f}.csv')
    
    print(f"✓ Matrices exported to {output_dir}")

