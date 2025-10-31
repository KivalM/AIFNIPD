#!/usr/bin/env python3
"""
Command-line interface for generating plots from tournament results.

Usage:
    uv run genplots results_dir [--no-latex] [--agent-groups GROUPS] [--rename-map MAP]
"""
import argparse
import sys
from pathlib import Path
import json
import numpy as np

from .result_utils import (
    load_all_agents_scores,
    load_all_agents_cc_rates,
    load_all_agents_cd_rates,
)
from .plot_utils import (
    setup_publication_style,
    filter_data_by_agents,
    plot_scores_vs_noise,
    plot_cc_rate_vs_noise,
    plot_cd_rate_vs_noise,
    create_comparison_plots,
    generate_plots_for_agent_groups,
    create_summary_table,
    create_formatted_table,
    create_ranking_table,
)


def parse_agent_groups(groups_str: str) -> list:
    """
    Parse agent groups from command line string.
    
    Expected format: "[[0,1,2],[3,4,5]]" or "0,1,2" for single group
    """
    try:
        # Try parsing as JSON first
        groups = json.loads(groups_str)
        if isinstance(groups[0], list):
            return groups
        else:
            return [groups]
    except (json.JSONDecodeError, TypeError):
        # Fall back to comma-separated
        indices = [int(x.strip()) for x in groups_str.split(',')]
        return [indices]


def parse_rename_map(map_str: str) -> dict:
    """
    Parse rename map from command line string.
    
    Expected format: '{"0": "AIF-R", "1": "AIF-C"}'
    """
    return json.loads(map_str)


def detect_pool_structure(results_dir: Path) -> tuple[bool, list]:
    """
    Detect if results directory has static/learning pool structure.
    
    Returns:
        Tuple of (has_pools, subdirs)
    """
    static_dir = results_dir / 'static'
    learning_dir = results_dir / 'learning'
    
    if static_dir.exists() and learning_dir.exists():
        return True, ['static', 'learning']
    else:
        return False, ['.']


def generate_all_plots(
    results_dir: Path,
    use_latex: bool = True,
    agent_groups: list = None,
    rename_maps: list = None,
    noise_levels: list = None
):
    """
    Generate all plots for a results directory.
    
    Args:
        results_dir: Path to results directory
        use_latex: Whether to use LaTeX for text rendering
        agent_groups: List of agent group lists for separate plots
        rename_maps: List of rename maps for agent groups
        noise_levels: List of noise levels to analyze
    """
    results_dir = Path(results_dir)
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Setup plotting style
    setup_publication_style(use_latex=use_latex)
    
    # Detect structure
    has_pools, subdirs = detect_pool_structure(results_dir)
    
    # Auto-detect noise levels if not provided
    if noise_levels is None:
        # Common noise levels used in experiments
        noise_levels = list(np.arange(0, 0.30, 0.05).round(2))
    
    print(f"Generating plots for: {results_dir}")
    print(f"Output directory: {plots_dir}")
    print(f"Pool structure: {'Static/Learning' if has_pools else 'Single pool'}")
    print(f"Noise levels: {noise_levels}")
    print()
    
    if has_pools:
        # Load data for both pools
        print("Loading data...")
        static_dir = results_dir / 'static'
        learning_dir = results_dir / 'learning'
        
        static_scores = load_all_agents_scores(static_dir, noise_levels)
        static_cc_rates = load_all_agents_cc_rates(static_dir, noise_levels)
        static_cd_rates = load_all_agents_cd_rates(static_dir, noise_levels)
        
        learning_scores = load_all_agents_scores(learning_dir, noise_levels)
        learning_cc_rates = load_all_agents_cc_rates(learning_dir, noise_levels)
        learning_cd_rates = load_all_agents_cd_rates(learning_dir, noise_levels)
        
        print(f"Static pool: {len(static_scores['agent_name'].unique())} agents")
        print(f"Learning pool: {len(learning_scores['agent_name'].unique())} agents")
        print()
        
        # Filter and rename if requested
        if agent_groups or rename_maps:
            if agent_groups:
                # For simplicity, apply first group's filters to all data
                agent_indices = agent_groups[0]
                print(f"Filtering agents: {agent_indices}")
            else:
                agent_indices = None
            
            rename_map = rename_maps[0] if rename_maps else None
            if rename_map:
                print(f"Renaming agents: {rename_map}")
            
            static_scores = filter_data_by_agents(static_scores, agent_indices, rename_map)
            static_cc_rates = filter_data_by_agents(static_cc_rates, agent_indices, rename_map)
            static_cd_rates = filter_data_by_agents(static_cd_rates, agent_indices, rename_map)
            
            learning_scores = filter_data_by_agents(learning_scores, agent_indices, rename_map)
            learning_cc_rates = filter_data_by_agents(learning_cc_rates, agent_indices, rename_map)
            learning_cd_rates = filter_data_by_agents(learning_cd_rates, agent_indices, rename_map)
            print()
        
        # Generate individual pool plots
        print("\nGenerating static pool plots...")
        plot_scores_vs_noise(
            static_scores,
            title='Static Pool: Score vs Noise',
            save_path=plots_dir / 'static_score_vs_noise'
        )
        plot_cc_rate_vs_noise(
            static_cc_rates,
            title='Static Pool: CC Rate vs Noise',
            save_path=plots_dir / 'static_cc_rate_vs_noise'
        )
        plot_cd_rate_vs_noise(
            static_cd_rates,
            title='Static Pool: CD Rate vs Noise',
            save_path=plots_dir / 'static_cd_rate_vs_noise'
        )
        
        print("Generating learning pool plots...")
        plot_scores_vs_noise(
            learning_scores,
            title='Learning Pool: Score vs Noise',
            save_path=plots_dir / 'learning_score_vs_noise'
        )
        plot_cc_rate_vs_noise(
            learning_cc_rates,
            title='Learning Pool: CC Rate vs Noise',
            save_path=plots_dir / 'learning_cc_rate_vs_noise'
        )
        plot_cd_rate_vs_noise(
            learning_cd_rates,
            title='Learning Pool: CD Rate vs Noise',
            save_path=plots_dir / 'learning_cd_rate_vs_noise'
        )
        
        # Generate summary tables
        print("\nGenerating summary tables...")
        static_summary = create_summary_table(
            static_scores, static_cc_rates, static_cd_rates, 'Static'
        )
        learning_summary = create_summary_table(
            learning_scores, learning_cc_rates, learning_cd_rates, 'Learning'
        )
        
        import pandas as pd
        combined_summary = pd.concat([static_summary, learning_summary], ignore_index=True)
        combined_summary.to_csv(plots_dir / 'comprehensive_summary.csv', index=False)
        
        formatted_summary = create_formatted_table(combined_summary)
        formatted_summary.to_csv(plots_dir / 'formatted_summary.csv', index=False)
        
        # Generate ranking tables
        static_rankings = create_ranking_table(static_scores, 'Static')
        learning_rankings = create_ranking_table(learning_scores, 'Learning')
        rankings = pd.concat([static_rankings, learning_rankings], ignore_index=True)
        rankings.to_csv(plots_dir / 'agent_rankings.csv', index=False)
        
        print(f"  - comprehensive_summary.csv")
        print(f"  - formatted_summary.csv")
        print(f"  - agent_rankings.csv")
        
    else:
        # Single pool structure
        print("Loading data...")
        scores = load_all_agents_scores(results_dir, noise_levels)
        cc_rates = load_all_agents_cc_rates(results_dir, noise_levels)
        cd_rates = load_all_agents_cd_rates(results_dir, noise_levels)
        
        print(f"Agents: {len(scores['agent_name'].unique())}")
        print()
        
        # Generate plots with agent groups if specified
        if agent_groups and len(agent_groups) > 1:
            print(f"Generating plots for {len(agent_groups)} agent groups...")
            
            generate_plots_for_agent_groups(
                scores, agent_groups, plot_scores_vs_noise,
                rename_maps=rename_maps,
                base_save_path=plots_dir / 'score_vs_noise',
                title='Score vs Noise'
            )
            
            generate_plots_for_agent_groups(
                cc_rates, agent_groups, plot_cc_rate_vs_noise,
                rename_maps=rename_maps,
                base_save_path=plots_dir / 'cc_rate_vs_noise',
                title='CC Rate vs Noise'
            )
            
            generate_plots_for_agent_groups(
                cd_rates, agent_groups, plot_cd_rate_vs_noise,
                rename_maps=rename_maps,
                base_save_path=plots_dir / 'cd_rate_vs_noise',
                title='CD Rate vs Noise'
            )
        else:
            # Single plot with all agents (or filtered)
            if agent_groups:
                agent_indices = agent_groups[0]
                rename_map = rename_maps[0] if rename_maps else None
                scores = filter_data_by_agents(scores, agent_indices, rename_map)
                cc_rates = filter_data_by_agents(cc_rates, agent_indices, rename_map)
                cd_rates = filter_data_by_agents(cd_rates, agent_indices, rename_map)
            
            print("Generating plots...")
            plot_scores_vs_noise(
                scores,
                title='Score vs Noise',
                save_path=plots_dir / 'score_vs_noise'
            )
            plot_cc_rate_vs_noise(
                cc_rates,
                title='CC Rate vs Noise',
                save_path=plots_dir / 'cc_rate_vs_noise'
            )
            plot_cd_rate_vs_noise(
                cd_rates,
                title='CD Rate vs Noise',
                save_path=plots_dir / 'cd_rate_vs_noise'
            )
        
        # Generate summary tables
        print("\nGenerating summary tables...")
        summary = create_summary_table(scores, cc_rates, cd_rates, 'Main')
        summary.to_csv(plots_dir / 'comprehensive_summary.csv', index=False)
        
        formatted_summary = create_formatted_table(summary)
        formatted_summary.to_csv(plots_dir / 'formatted_summary.csv', index=False)
        
        rankings = create_ranking_table(scores, 'Main')
        rankings.to_csv(plots_dir / 'agent_rankings.csv', index=False)
    
    print("\n" + "="*60)
    print(f"✓ All plots saved to: {plots_dir}")
    print("="*60)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='Generate publication-quality plots from tournament results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all plots for a results directory
  uv run genplots results_1000/main

  # Generate plots without LaTeX (faster, no LaTeX installation needed)
  uv run genplots results_1000/main --no-latex

  # Generate plots for specific agent groups
  uv run genplots results_1000/main --agent-groups "[[0,1,2],[3,4,5]]"

  # Filter and rename agents
  uv run genplots results_1000/main --agent-groups "0,1,2,3" --rename-map '{"0":"AIF-R","1":"AIF-C"}'

  # Specify custom noise levels
  uv run genplots results_1000/main --noise-levels "0,0.1,0.2,0.3"
        """
    )
    
    parser.add_argument(
        'results_dir',
        type=Path,
        help='Path to results directory (e.g., results_1000/main)'
    )
    
    parser.add_argument(
        '--no-latex',
        action='store_true',
        help='Disable LaTeX text rendering (useful if LaTeX not installed)'
    )
    
    parser.add_argument(
        '--agent-groups',
        type=str,
        help='Agent groups to plot separately, as JSON list of lists: "[[0,1,2],[3,4,5]]" or single group: "0,1,2"'
    )
    
    parser.add_argument(
        '--rename-map',
        type=str,
        help='JSON mapping to rename agents: \'{"0":"AIF-R","1":"AIF-C"}\''
    )
    
    parser.add_argument(
        '--noise-levels',
        type=str,
        help='Comma-separated noise levels: "0,0.05,0.1,0.15,0.2,0.25"'
    )
    
    args = parser.parse_args()
    
    # Validate results directory
    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Parse optional arguments
    agent_groups = None
    if args.agent_groups:
        try:
            agent_groups = parse_agent_groups(args.agent_groups)
        except Exception as e:
            print(f"Error parsing agent groups: {e}", file=sys.stderr)
            sys.exit(1)
    
    rename_maps = None
    if args.rename_map:
        try:
            rename_map = parse_rename_map(args.rename_map)
            # If we have agent groups, create a rename map for each group
            # For simplicity, use the same map for all groups
            if agent_groups:
                rename_maps = [rename_map] * len(agent_groups)
            else:
                rename_maps = [rename_map]
        except Exception as e:
            print(f"Error parsing rename map: {e}", file=sys.stderr)
            sys.exit(1)
    
    noise_levels = None
    if args.noise_levels:
        try:
            noise_levels = [float(x.strip()) for x in args.noise_levels.split(',')]
        except Exception as e:
            print(f"Error parsing noise levels: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Generate plots
    try:
        generate_all_plots(
            args.results_dir,
            use_latex=not args.no_latex,
            agent_groups=agent_groups,
            rename_maps=rename_maps,
            noise_levels=noise_levels
        )
    except Exception as e:
        print(f"\nError generating plots: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

