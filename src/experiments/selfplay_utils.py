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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate pairwise CC rate, CD rate, score, and normalized cooperation matrices from tournament results.
    
    Args:
        df: DataFrame with tournament results
        player_names: List of player names in order
        
    Returns:
        Tuple of (cc_rate_matrix, cd_rate_matrix, score_matrix, norm_coop_matrix)
        Each matrix is N×N where N is the number of players
    """
    n_players = len(player_names)
    cc_rate_matrix = np.zeros((n_players, n_players))
    cd_rate_matrix = np.zeros((n_players, n_players))
    score_matrix = np.zeros((n_players, n_players))
    norm_coop_matrix = np.zeros((n_players, n_players))
    norm_coop_matrix[:] = np.nan  # Initialize with NaN to handle division by zero
    
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
                
                # Calculate normalized cooperation (CC / (CC + CD))
                total_coop = cc_count + cd_count
                if total_coop > 0:
                    norm_coop_matrix[player_idx, opponent_idx] = cc_count / total_coop
                # If total_coop == 0, leave as NaN
            
            # Mean score
            score_matrix[player_idx, opponent_idx] = matchup_df['Score'].mean()
    
    return cc_rate_matrix, cd_rate_matrix, score_matrix, norm_coop_matrix


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
        cc_matrix, cd_matrix, score_matrix, norm_coop_matrix = calculate_pairwise_metrics(df, player_names)
        
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
        
        # Normalized Cooperation heatmap
        print(f"  Creating normalized cooperation heatmap...")
        plot_heatmap(
            norm_coop_matrix,
            labels,
            f'Normalized Cooperation (CC/(CC+CD)) at Noise {noise:.2f}',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            cbar_label='Normalized Cooperation',
            figsize=figsize,
            annot=True,
            fmt='.3f',
            save_path=output_dir / f'norm_coop_noise_{noise:.2f}.pdf'
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
    elif metric == 'norm_coop':
        cmap = 'RdYlGn'
        vmin, vmax = 0, 1
        cbar_label = 'Normalized Cooperation'
        title_prefix = 'Normalized Cooperation'
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
        cc_matrix, cd_matrix, score_matrix, norm_coop_matrix = calculate_pairwise_metrics(df, player_names)
        
        # Select the right matrix
        if metric == 'cc_rate':
            matrix = cc_matrix
        elif metric == 'cd_rate':
            matrix = cd_matrix
        elif metric == 'norm_coop':
            matrix = norm_coop_matrix
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
        
        cc_matrix, cd_matrix, score_matrix, norm_coop_matrix = calculate_pairwise_metrics(df, player_names)
        
        # Convert to DataFrames with labels
        cc_df = pd.DataFrame(cc_matrix, index=labels, columns=labels)
        cd_df = pd.DataFrame(cd_matrix, index=labels, columns=labels)
        score_df = pd.DataFrame(score_matrix, index=labels, columns=labels)
        norm_coop_df = pd.DataFrame(norm_coop_matrix, index=labels, columns=labels)
        
        # Save to CSV
        cc_df.to_csv(output_dir / f'cc_rate_noise_{noise:.2f}.csv')
        cd_df.to_csv(output_dir / f'cd_rate_noise_{noise:.2f}.csv')
        score_df.to_csv(output_dir / f'score_noise_{noise:.2f}.csv')
        norm_coop_df.to_csv(output_dir / f'norm_coop_noise_{noise:.2f}.csv')
    
    print(f"✓ Matrices exported to {output_dir}")


def generate_selfplay_comparison_tables(
    sp_dir: Path,
    output_dir: Path,
    rename_map: Optional[Dict[int, str]] = None,
    noise_levels: List[float] = [0.0, 0.05]
):
    """
    Generate formatted comparison tables for self-play tournament results.
    
    Creates tables comparing agent performance at different noise levels,
    showing mean ± std (IQR) format similar to main tournament analysis.
    
    Args:
        sp_dir: Path to self-play results directory
        output_dir: Directory to save comparison tables
        rename_map: Optional mapping from agent index to display name
        noise_levels: List of noise levels to compare (default: [0.0, 0.05])
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get player names
    player_names = get_player_names(sp_dir)
    if not player_names:
        print("No player names found!")
        return
    
    n_players = len(player_names)
    labels = create_agent_labels(player_names, rename_map)
    
    # Prepare data structures for each metric
    metrics = {
        'cc_rate': {'title': 'CC Rate', 'filename': 'selfplay_cc_rate_comparison.csv'},
        'norm_coop': {'title': 'Normalized Cooperation', 'filename': 'selfplay_norm_coop_comparison.csv'},
        'cd_rate': {'title': 'CD Rate', 'filename': 'selfplay_cd_rate_comparison.csv'},
        'score': {'title': 'Score', 'filename': 'selfplay_score_comparison.csv'}
    }
    
    for metric_key, metric_info in metrics.items():
        print(f"  Generating {metric_info['title']} comparison table...")
        
        comparison_data = []
        
        for player_idx, (player_name, label) in enumerate(zip(player_names, labels)):
            # Collect values for each noise level
            noise_data = {}
            
            for noise in noise_levels:
                df = load_selfplay_results(sp_dir, noise)
                if df.empty:
                    continue
                
                # Calculate metrics
                cc_matrix, cd_matrix, score_matrix, norm_coop_matrix = calculate_pairwise_metrics(df, player_names)
                
                # Select the appropriate matrix
                if metric_key == 'cc_rate':
                    matrix = cc_matrix
                elif metric_key == 'norm_coop':
                    matrix = norm_coop_matrix
                elif metric_key == 'cd_rate':
                    matrix = cd_matrix
                else:  # score
                    matrix = score_matrix
                
                # Get values for this agent against all opponents (excluding self)
                agent_values = []
                for opp_idx in range(n_players):
                    if opp_idx != player_idx:  # Exclude self-play
                        value = matrix[player_idx, opp_idx]
                        if not np.isnan(value):
                            agent_values.append(value)
                
                if agent_values:
                    values_arr = np.array(agent_values)
                    noise_data[noise] = {
                        'mean': np.mean(values_arr),
                        'std': np.std(values_arr, ddof=1) if len(values_arr) > 1 else 0,
                        'q25': np.percentile(values_arr, 25),
                        'q75': np.percentile(values_arr, 75),
                        'iqr': np.percentile(values_arr, 75) - np.percentile(values_arr, 25)
                    }
            
            # If we have data for both noise levels, create comparison entry
            if len(noise_data) == len(noise_levels):
                noise_0_data = noise_data[noise_levels[0]]
                noise_1_data = noise_data[noise_levels[1]]
                
                comparison_data.append({
                    'Agent': label,
                    f'{metric_info["title"]}_{noise_levels[0]:.2f}': noise_0_data['mean'],
                    f'StdDev_{noise_levels[0]:.2f}': noise_0_data['std'],
                    f'IQR_{noise_levels[0]:.2f}': noise_0_data['iqr'],
                    f'{metric_info["title"]}_{noise_levels[1]:.2f}': noise_1_data['mean'],
                    f'StdDev_{noise_levels[1]:.2f}': noise_1_data['std'],
                    f'IQR_{noise_levels[1]:.2f}': noise_1_data['iqr'],
                    'Difference': noise_1_data['mean'] - noise_0_data['mean']
                })
        
        if comparison_data:
            # Create DataFrame
            df = pd.DataFrame(comparison_data)
            df = df.sort_values('Agent')
            
            # Save raw version
            raw_filename = metric_info['filename'].replace('.csv', '_raw.csv')
            df.to_csv(output_dir / raw_filename, index=False)
            
            # Create formatted version
            formatted_data = []
            for _, row in df.iterrows():
                # Construct column names
                col_0_name = f'{metric_info["title"]}_{noise_levels[0]:.2f}'
                col_1_name = f'{metric_info["title"]}_{noise_levels[1]:.2f}'
                std_0_name = f'StdDev_{noise_levels[0]:.2f}'
                std_1_name = f'StdDev_{noise_levels[1]:.2f}'
                iqr_0_name = f'IQR_{noise_levels[0]:.2f}'
                iqr_1_name = f'IQR_{noise_levels[1]:.2f}'
                
                formatted_data.append({
                    'Agent': row['Agent'],
                    col_0_name: 
                        f"{row[col_0_name]:.4f} ± "
                        f"{row[std_0_name]:.4f} "
                        f"(IQR: {row[iqr_0_name]:.4f})",
                    col_1_name: 
                        f"{row[col_1_name]:.4f} ± "
                        f"{row[std_1_name]:.4f} "
                        f"(IQR: {row[iqr_1_name]:.4f})",
                    'Difference': f"{row['Difference']:+.4f}"
                })
            
            formatted_df = pd.DataFrame(formatted_data)
            formatted_df.to_csv(output_dir / metric_info['filename'], index=False)
            
            print(f"    ✓ Saved {metric_info['filename']}")
    
    print(f"  ✓ All comparison tables saved to {output_dir}")


def generate_selfplay_diagonal_comparison_tables(
    sp_dir: Path,
    output_dir: Path,
    rename_map: Optional[Dict[int, str]] = None,
    noise_levels: List[float] = [0.0, 0.05]
):
    """
    Generate formatted comparison tables for self-play diagonal (agent vs itself).
    
    Creates tables showing agent performance when playing against itself,
    showing mean ± std (IQR) format across repetitions.
    
    Args:
        sp_dir: Path to self-play results directory
        output_dir: Directory to save comparison tables
        rename_map: Optional mapping from agent index to display name
        noise_levels: List of noise levels to compare (default: [0.0, 0.05])
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get player names
    player_names = get_player_names(sp_dir)
    if not player_names:
        print("No player names found!")
        return
    
    n_players = len(player_names)
    labels = create_agent_labels(player_names, rename_map)
    
    # Prepare data structures for each metric
    metrics = {
        'cc_rate': {'title': 'CC Rate', 'filename': 'selfplay_diagonal_cc_rate_comparison.csv'},
        'norm_coop': {'title': 'Normalized Cooperation', 'filename': 'selfplay_diagonal_norm_coop_comparison.csv'},
        'cd_rate': {'title': 'CD Rate', 'filename': 'selfplay_diagonal_cd_rate_comparison.csv'},
        'score': {'title': 'Score', 'filename': 'selfplay_diagonal_score_comparison.csv'}
    }
    
    for metric_key, metric_info in metrics.items():
        print(f"  Generating diagonal {metric_info['title']} comparison table...")
        
        comparison_data = []
        
        for player_idx, (player_name, label) in enumerate(zip(player_names, labels)):
            # Collect diagonal values (self-play) for each noise level
            noise_data = {}
            
            for noise in noise_levels:
                df = load_selfplay_results(sp_dir, noise)
                if df.empty:
                    continue
                
                # Filter for self-play matches (player vs itself)
                self_play_df = df[
                    (df['Player name'] == player_name) & 
                    (df['Opponent name'] == player_name)
                ]
                
                if self_play_df.empty:
                    continue
                
                # Calculate metric per repetition
                rep_values = []
                for rep in self_play_df['repetition'].unique():
                    rep_data = self_play_df[self_play_df['repetition'] == rep]
                    
                    if metric_key == 'score':
                        rep_values.append(rep_data['Score'].mean())
                    else:
                        total_turns = rep_data['Turns'].sum()
                        if total_turns > 0:
                            if metric_key == 'cc_rate':
                                cc_count = rep_data['CC count'].sum()
                                rep_values.append(cc_count / total_turns)
                            elif metric_key == 'cd_rate':
                                cd_count = rep_data['CD count'].sum()
                                rep_values.append(cd_count / total_turns)
                            elif metric_key == 'norm_coop':
                                cc_count = rep_data['CC count'].sum()
                                cd_count = rep_data['CD count'].sum()
                                total_coop = cc_count + cd_count
                                if total_coop > 0:
                                    rep_values.append(cc_count / total_coop)
                
                if rep_values:
                    values_arr = np.array(rep_values)
                    noise_data[noise] = {
                        'mean': np.mean(values_arr),
                        'std': np.std(values_arr, ddof=1) if len(values_arr) > 1 else 0,
                        'q25': np.percentile(values_arr, 25),
                        'q75': np.percentile(values_arr, 75),
                        'iqr': np.percentile(values_arr, 75) - np.percentile(values_arr, 25)
                    }
            
            # If we have data for both noise levels, create comparison entry
            if len(noise_data) == len(noise_levels):
                noise_0_data = noise_data[noise_levels[0]]
                noise_1_data = noise_data[noise_levels[1]]
                
                comparison_data.append({
                    'Agent': label,
                    f'{metric_info["title"]}_{noise_levels[0]:.2f}': noise_0_data['mean'],
                    f'StdDev_{noise_levels[0]:.2f}': noise_0_data['std'],
                    f'IQR_{noise_levels[0]:.2f}': noise_0_data['iqr'],
                    f'{metric_info["title"]}_{noise_levels[1]:.2f}': noise_1_data['mean'],
                    f'StdDev_{noise_levels[1]:.2f}': noise_1_data['std'],
                    f'IQR_{noise_levels[1]:.2f}': noise_1_data['iqr'],
                    'Difference': noise_1_data['mean'] - noise_0_data['mean']
                })
        
        if comparison_data:
            # Create DataFrame
            df = pd.DataFrame(comparison_data)
            df = df.sort_values('Agent')
            
            # Save raw version
            raw_filename = metric_info['filename'].replace('.csv', '_raw.csv')
            df.to_csv(output_dir / raw_filename, index=False)
            
            # Create formatted version
            formatted_data = []
            for _, row in df.iterrows():
                # Construct column names
                col_0_name = f'{metric_info["title"]}_{noise_levels[0]:.2f}'
                col_1_name = f'{metric_info["title"]}_{noise_levels[1]:.2f}'
                std_0_name = f'StdDev_{noise_levels[0]:.2f}'
                std_1_name = f'StdDev_{noise_levels[1]:.2f}'
                iqr_0_name = f'IQR_{noise_levels[0]:.2f}'
                iqr_1_name = f'IQR_{noise_levels[1]:.2f}'
                
                formatted_data.append({
                    'Agent': row['Agent'],
                    col_0_name: 
                        f"{row[col_0_name]:.4f} ± "
                        f"{row[std_0_name]:.4f} "
                        f"(IQR: {row[iqr_0_name]:.4f})",
                    col_1_name: 
                        f"{row[col_1_name]:.4f} ± "
                        f"{row[std_1_name]:.4f} "
                        f"(IQR: {row[iqr_1_name]:.4f})",
                    'Difference': f"{row['Difference']:+.4f}"
                })
            
            formatted_df = pd.DataFrame(formatted_data)
            formatted_df.to_csv(output_dir / metric_info['filename'], index=False)
            
            print(f"    ✓ Saved {metric_info['filename']}")
    
    print(f"  ✓ All diagonal comparison tables saved to {output_dir}")

