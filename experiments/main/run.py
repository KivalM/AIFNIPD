from typing import Any
from tournament.handler import FileSystemHandler
from tournament.tournament import NoiseTournament
from experiments import static_pool, learn_pool
import itertools
import multiprocessing
from agents.aif.jax.five_state import JaxFiveStateAgent
from agents.aif.jax.five_state_noise import JaxFiveStateAgentNoisy
from agents.aif.jax.five_state_utility import JaxFiveStateAgentUtility
from agents.aif.jax.five_state_deterministic import JaxFiveStateAgentDeterministic
from agents.bqlearner import JaxBayesianQLearner, CooperativeBQLearner
from agents.qlearner import JaxQLearner, CooperativeQLearner
from agents.psrl import PSRL, CooperativePSRL
from agents.dynaQ import DynaQ, CooperativeDynaQ
import axelrod as axl
import numpy as np
import os
import itertools
import random
import tqdm
from pathlib import Path
from experiments import (
    load_all_agents_scores,
    load_all_agents_cc_rates,
    load_all_agents_cd_rates,
    load_all_agents_normalized_cooperation,
    setup_publication_style,
    filter_data_by_agents,
    plot_scores_vs_noise,
    plot_cc_rate_vs_noise,
    plot_cd_rate_vs_noise,
    plot_normalized_cooperation_vs_noise,
    load_agent_repetition_scores,
    compare_variances_pairwise,
    compare_variances_all,
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Environment parameters
noise_levels = list(np.arange(0, 0.30, 0.05).round(2))
repetitions = 30
turns = 1000
processes = 20
seed = 42

strategies = [
    JaxFiveStateAgent(
        pB_scale=1,
        gamma=1,
        alpha=1,
        bias=0.5,
        preference="standard",
        policy_len=5,
        update_interval=5,
        seed=seed,
        lr_B=1,
    ),
    JaxFiveStateAgent(
        pB_scale=1,
        gamma=1,
        alpha=1,
        bias=0.5,
        preference="nash",
        policy_len=10,
        update_interval=50,
        seed=seed,
        lr_B=1,
    ),
    JaxQLearner(
        learning_rate=0.9,
        discount_rate=0.9,
        action_selection_parameter=0.1,
    ),
    CooperativeQLearner(
        learning_rate=0.9,
        discount_rate=0.9,
        action_selection_parameter=0.1,
    ),

    JaxBayesianQLearner(
        discount_rate=0.5,
        initial_variance=1.0,
        reward_variance=1.0,
    ),
    CooperativeBQLearner(
        discount_rate=0.5,
        initial_variance=1.0,
        reward_variance=1.0,
    ),
    JaxFiveStateAgentNoisy(
        pB_scale=1,
        gamma=1,
        alpha=1,
        bias=0.5,
        preference="standard",
        policy_len=10,
        update_interval=50,
        seed=seed,
        lr_B=1,
    ),
    JaxFiveStateAgentNoisy(
        pB_scale=1,
        gamma=1,
        alpha=1,
        bias=0.5,
        preference="nash",
        policy_len=10,
        update_interval=50,
        seed=seed,
        lr_B=1,
    ),
    #
    axl.DBS(0.999, 5, 2, 3, 5),
    axl.GTFT(),
    axl.ContriteTitForTat(),
    JaxFiveStateAgentUtility(
        pB_scale=1,
        gamma=1,
        alpha=1,
        bias=0.5,
        preference="standard",
        policy_len=10,
        update_interval=50,
        seed=seed,
        lr_B=1,
    ),
    JaxFiveStateAgentUtility(
        pB_scale=1,
        gamma=1,
        alpha=1,
        bias=0.5,
        preference="nash",
        policy_len=10,
        update_interval=50,
        seed=seed,
        lr_B=1,
    ),
    DynaQ(
        learning_rate=0.1,
        discount_rate=0.9,
        action_selection_parameter=0.1,
        planning_steps=5,
    ),
    CooperativeDynaQ(
        learning_rate=0.1,
        discount_rate=0.9,
        action_selection_parameter=0.1,
        planning_steps=5,
    ),
    PSRL(
        prior_strength=0.5,
        discount_rate=0.95,
        value_iteration_steps=50,
    ),
    CooperativePSRL(
        prior_strength=0.5,
        discount_rate=0.95,
        value_iteration_steps=50,
    ),
    JaxFiveStateAgentDeterministic(
        pB_scale=1,
        gamma=1,
        alpha=1,
        bias=0.5,
        preference="standard",
        policy_len=5,
        update_interval=5,
        seed=seed,
        lr_B=1,
    ),
    JaxFiveStateAgentDeterministic(
        pB_scale=1,
        gamma=1,
        alpha=1,
        bias=0.5,
        preference="nash",
        policy_len=5,
        update_interval=5,
        seed=seed,
        lr_B=1,
    ),
]

def run_experiment(strategies, opponents, dir_name):
    for i, strategy in enumerate[Any](strategies):
        handler = FileSystemHandler(root_dir=f"results/main/{dir_name}/{i}_{strategy.__class__.__name__}")
        noise_tournament = NoiseTournament(
            players=[strategy] + opponents,
            noise_levels=noise_levels,
            repetitions=repetitions,
            seed=seed,
            callback=handler.save_results,
            skip_callback=handler.skip_run,
        )
        noise_tournament.run(turns=turns, processes=processes)
        print(f"Results are saved in the '{handler.root_dir}' directory.")


def generate_plots():
    """Generate publication-quality plots after experiments complete."""
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60 + "\n")
    
    # Setup publication style
    setup_publication_style(use_latex=False)
    
    # Agent rename map based on indices
    rename_map = {
        '0': 'AIF-R',           # JaxFiveStateAgent (standard)
        '1': 'AIF-C',           # JaxFiveStateAgent (nash)
        '2': 'QL-R',            # JaxQLearner
        '3': 'QL-C',            # CooperativeQLearner
        '4': 'BQL-R',           # JaxBayesianQLearner
        '5': 'BQL-C',           # CooperativeBQLearner
        '6': 'AIF-R-N',         # JaxFiveStateAgentNoisy (standard)
        '7': 'AIF-C-N',         # JaxFiveStateAgentNoisy (nash)
        '8': 'DBS',             # DBS
        '9': 'GTFT',            # GTFT
        '10': 'CTFT',           # ContriteTitForTat
        '11': 'AIF-R-U',        # JaxFiveStateAgentUtility (standard)
        '12': 'AIF-C-U',        # JaxFiveStateAgentUtility (nash)
    }
    
    # Agent groups for focused plots
    # Group 1: Cooperative agents comparison
    cooperative_group = [7, 1, 5, 3, 8]  # AIF-C-N, AIF-C, BQL-C, QL-C, DBS
    
    # Group 2: Rational agents comparison
    rational_group = [6, 4, 2, 0, 8]  # AIF-R-N, BQL-R, QL-R, AIF-R, DBS
    
    # Group 3: Comparison across agent types
    mixed_group = [7, 6, 2, 3, 8]  # AIF-C-N, AIF-R-N, QL-R, QL-C, DBS
    
    # Create top-level plots and analysis directories
    main_plots_dir = Path('results/main/plots')
    main_plots_dir.mkdir(exist_ok=True)
    
    analysis_dir = Path('results/main/analysis')
    analysis_dir.mkdir(exist_ok=True)
    
    # Process both pools
    for pool_name in ['static', 'learning']:
        print(f"\nProcessing {pool_name} pool...")
        pool_dir = Path(f'results/main/{pool_name}')
        plots_dir = main_plots_dir / pool_name
        plots_dir.mkdir(exist_ok=True)
        
        # Load data
        print(f"  Loading data...")
        scores = load_all_agents_scores(pool_dir, noise_levels)
        cc_rates = load_all_agents_cc_rates(pool_dir, noise_levels)
        cd_rates = load_all_agents_cd_rates(pool_dir, noise_levels)
        norm_coop = load_all_agents_normalized_cooperation(pool_dir, noise_levels)
        
        # Remove any duplicate entries (shouldn't happen but just in case)
        scores = scores.drop_duplicates(subset=['agent_name', 'noise_level'], keep='first')
        cc_rates = cc_rates.drop_duplicates(subset=['agent_name', 'noise_level'], keep='first')
        cd_rates = cd_rates.drop_duplicates(subset=['agent_name', 'noise_level'], keep='first')
        norm_coop = norm_coop.drop_duplicates(subset=['agent_name', 'noise_level'], keep='first')
        
        # Generate plots for cooperative agents
        print(f"  Generating cooperative agents plots...")
        cooperative_scores = filter_data_by_agents(scores, cooperative_group, rename_map)
        cooperative_cc = filter_data_by_agents(cc_rates, cooperative_group, rename_map)
        cooperative_cd = filter_data_by_agents(cd_rates, cooperative_group, rename_map)
        cooperative_norm_coop = filter_data_by_agents(norm_coop, cooperative_group, rename_map)
        
        # Debug: print unique agents in the filtered data
        print(f"    Cooperative agents: {sorted(cooperative_scores['agent_name'].unique())}")
        
        plot_scores_vs_noise(
            cooperative_scores,
            title=f'{pool_name.capitalize()} Pool: Cooperative Agents',
            figsize=(10, 6),
            save_path=plots_dir / f'{pool_name}_cooperative_scores'
        )
        
        plot_cc_rate_vs_noise(
            cooperative_cc,
            title=f'{pool_name.capitalize()} Pool: Cooperative Agents CC Rate',
            figsize=(10, 6),
            save_path=plots_dir / f'{pool_name}_cooperative_cc_rate'
        )
        
        plot_normalized_cooperation_vs_noise(
            cooperative_norm_coop,
            title=f'{pool_name.capitalize()} Pool: Cooperative Agents Normalized Cooperation',
            figsize=(10, 6),
            save_path=plots_dir / f'{pool_name}_cooperative_norm_coop'
        )
        
        plot_cd_rate_vs_noise(
            cooperative_cd,
            title=f'{pool_name.capitalize()} Pool: Cooperative Agents CD Rate',
            figsize=(10, 6),
            save_path=plots_dir / f'{pool_name}_cooperative_cd_rate'
        )
        
        # Generate plots for rational agents
        print(f"  Generating rational agents plots...")
        rational_scores = filter_data_by_agents(scores, rational_group, rename_map)
        rational_cc = filter_data_by_agents(cc_rates, rational_group, rename_map)
        rational_cd = filter_data_by_agents(cd_rates, rational_group, rename_map)
        rational_norm_coop = filter_data_by_agents(norm_coop, rational_group, rename_map)
        
        # Debug: print unique agents in the filtered data
        print(f"    Rational agents: {sorted(rational_scores['agent_name'].unique())}")
        
        plot_scores_vs_noise(
            rational_scores,
            title=f'{pool_name.capitalize()} Pool: Rational Agents',
            figsize=(10, 6),
            save_path=plots_dir / f'{pool_name}_rational_scores'
        )
        
        plot_cc_rate_vs_noise(
            rational_cc,
            title=f'{pool_name.capitalize()} Pool: Rational Agents CC Rate',
            figsize=(10, 6),
            save_path=plots_dir / f'{pool_name}_rational_cc_rate'
        )
        
        plot_normalized_cooperation_vs_noise(
            rational_norm_coop,
            title=f'{pool_name.capitalize()} Pool: Rational Agents Normalized Cooperation',
            figsize=(10, 6),
            save_path=plots_dir / f'{pool_name}_rational_norm_coop'
        )
        
        plot_cd_rate_vs_noise(
            rational_cd,
            title=f'{pool_name.capitalize()} Pool: Rational Agents CD Rate',
            figsize=(10, 6),
            save_path=plots_dir / f'{pool_name}_rational_cd_rate'
        )
        
        # Generate plots for mixed group (AIF-C-N, AIF-R-N, QL-R, QL-C, DBS)
        print(f"  Generating mixed agents comparison plots...")
        mixed_scores = filter_data_by_agents(scores, mixed_group, rename_map)
        mixed_cc = filter_data_by_agents(cc_rates, mixed_group, rename_map)
        mixed_cd = filter_data_by_agents(cd_rates, mixed_group, rename_map)
        mixed_norm_coop = filter_data_by_agents(norm_coop, mixed_group, rename_map)
        
        # Debug: print unique agents in the filtered data
        print(f"    Mixed agents: {sorted(mixed_scores['agent_name'].unique())}")
        
        plot_scores_vs_noise(
            mixed_scores,
            title=f'{pool_name.capitalize()} Pool: Cross-Type Comparison',
            figsize=(10, 6),
            save_path=plots_dir / f'{pool_name}_mixed_scores'
        )
        
        plot_cc_rate_vs_noise(
            mixed_cc,
            title=f'{pool_name.capitalize()} Pool: Cross-Type Comparison CC Rate',
            figsize=(10, 6),
            save_path=plots_dir / f'{pool_name}_mixed_cc_rate'
        )
        
        plot_normalized_cooperation_vs_noise(
            mixed_norm_coop,
            title=f'{pool_name.capitalize()} Pool: Cross-Type Comparison Normalized Cooperation',
            figsize=(10, 6),
            save_path=plots_dir / f'{pool_name}_mixed_norm_coop'
        )
        
        plot_cd_rate_vs_noise(
            mixed_cd,
            title=f'{pool_name.capitalize()} Pool: Cross-Type Comparison CD Rate',
            figsize=(10, 6),
            save_path=plots_dir / f'{pool_name}_mixed_cd_rate'
        )
        
        print(f"  ✓ Plots saved to {plots_dir}")
    
    print("\n" + "="*60)
    print("Generating analysis tables...")
    print("="*60)
    
    # Generate CC-Rate comparison table at noise 0 vs 0.05
    generate_cc_rate_comparison_table(main_plots_dir, analysis_dir, rename_map)
    
    # Generate variance analysis for DBS, AIF-C, and QL-C
    generate_variance_analysis(analysis_dir, rename_map)
    
    # Generate variance visualizations
    visualize_variance_analysis(analysis_dir, main_plots_dir)
    
    # Generate CVaR analysis
    generate_cvar_analysis(main_plots_dir, rename_map)
    
    # Generate comprehensive analysis CSV
    generate_comprehensive_analysis_csv(analysis_dir, rename_map)
    
    print("\n" + "="*60)
    print("Plot generation complete!")
    print("="*60)


def generate_cc_rate_comparison_table(main_plots_dir: Path, analysis_dir: Path, rename_map: dict):
    """Generate table comparing CC rates at noise 0 vs 0.05."""
    print("\n  Creating CC-Rate comparison table (noise 0 vs 0.05)...")
    
    comparison_data = []
    
    for pool_name in ['static', 'learning']:
        pool_dir = Path(f'results/main/{pool_name}')
        
        # Load CC rates for both noise levels
        cc_rates = load_all_agents_cc_rates(pool_dir, [0.0, 0.05])
        
        for agent_dir in cc_rates['agent_name'].unique():
            # Get agent index and rename
            agent_index = agent_dir.split('_')[0]
            agent_name = rename_map.get(agent_index, agent_dir)
            
            # Get data for each noise level
            noise_0 = cc_rates[(cc_rates['agent_name'] == agent_dir) & 
                               (cc_rates['noise_level'] == 0.0)]
            noise_005 = cc_rates[(cc_rates['agent_name'] == agent_dir) & 
                                 (cc_rates['noise_level'] == 0.05)]
            
            if not noise_0.empty and not noise_005.empty:
                # Load repetition data to calculate IQR
                from experiments.result_utils import load_tournament_results
                
                # Calculate per-repetition CC rates for noise 0
                df_0 = load_tournament_results(pool_dir, agent_dir, 0.0)
                if not df_0.empty:
                    agent_data_0 = df_0[df_0['Player index'] == 0]
                    cc_rates_0 = []
                    for rep in agent_data_0['repetition'].unique():
                        rep_data = agent_data_0[agent_data_0['repetition'] == rep]
                        total_turns = rep_data['Turns'].sum()
                        cc_count = rep_data['CC count'].sum()
                        if total_turns > 0:
                            cc_rates_0.append(cc_count / total_turns)
                    
                    # Calculate per-repetition CC rates for noise 0.05
                    df_005 = load_tournament_results(pool_dir, agent_dir, 0.05)
                    agent_data_005 = df_005[df_005['Player index'] == 0]
                    cc_rates_005 = []
                    for rep in agent_data_005['repetition'].unique():
                        rep_data = agent_data_005[agent_data_005['repetition'] == rep]
                        total_turns = rep_data['Turns'].sum()
                        cc_count = rep_data['CC count'].sum()
                        if total_turns > 0:
                            cc_rates_005.append(cc_count / total_turns)
                    
                    if cc_rates_0 and cc_rates_005:
                        comparison_data.append({
                            'Pool': pool_name.capitalize(),
                            'Agent': agent_name,
                            'CC_Rate_0.0': noise_0['mean_cc_rate'].values[0],
                            'StdDev_0.0': noise_0['std_cc_rate'].values[0],
                            'IQR_0.0': np.percentile(cc_rates_0, 75) - np.percentile(cc_rates_0, 25),
                            'CC_Rate_0.05': noise_005['mean_cc_rate'].values[0],
                            'StdDev_0.05': noise_005['std_cc_rate'].values[0],
                            'IQR_0.05': np.percentile(cc_rates_005, 75) - np.percentile(cc_rates_005, 25),
                            'Difference': noise_005['mean_cc_rate'].values[0] - noise_0['mean_cc_rate'].values[0]
                        })
    
    # Create DataFrame and save
    df = pd.DataFrame(comparison_data)
    df = df.sort_values(['Pool', 'Agent'])
    df.to_csv(analysis_dir / 'cc_rate_comparison_0_vs_005.csv', index=False)
    
    # Create formatted version
    formatted_data = []
    for _, row in df.iterrows():
        formatted_data.append({
            'Pool': row['Pool'],
            'Agent': row['Agent'],
            'CC_Rate_0.0': f"{row['CC_Rate_0.0']:.4f} ± {row['StdDev_0.0']:.4f} (IQR: {row['IQR_0.0']:.4f})",
            'CC_Rate_0.05': f"{row['CC_Rate_0.05']:.4f} ± {row['StdDev_0.05']:.4f} (IQR: {row['IQR_0.05']:.4f})",
            'Difference': f"{row['Difference']:+.4f}"
        })
    
    formatted_df = pd.DataFrame(formatted_data)
    formatted_df.to_csv(analysis_dir / 'cc_rate_comparison_0_vs_005_formatted.csv', index=False)
    
    print(f"    ✓ Saved to {analysis_dir / 'cc_rate_comparison_0_vs_005.csv'}")
    print(f"    ✓ Saved to {analysis_dir / 'cc_rate_comparison_0_vs_005_formatted.csv'}")


def generate_variance_analysis(analysis_dir: Path, rename_map: dict):
    """Analyze variance differences between DBS, AIF-C, and QL-C."""
    print("\n  Analyzing variance for DBS, AIF-C, and QL-C...")
    
    # Agents to compare: DBS (8), AIF-C (1), QL-C (3)
    agent_indices = {
        '8': 'DBS',
        '1': 'AIF-C', 
        '3': 'QL-C'
    }
    
    variance_results = []
    
    for pool_name in ['static', 'learning']:
        pool_dir = Path(f'results/main/{pool_name}')
        
        print(f"    Processing {pool_name} pool...")
        
        # Analyze at multiple noise levels
        for noise in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
            # Load repetition scores for each agent
            data_dict = {}
            for agent_idx, agent_name in agent_indices.items():
                agent_dir = f"{agent_idx}_{''}"  # Need to find full dir name
                # Find the actual directory
                for d in pool_dir.iterdir():
                    if d.is_dir() and d.name.startswith(f"{agent_idx}_"):
                        scores = load_agent_repetition_scores(pool_dir, d.name, noise)
                        if len(scores) > 0:
                            data_dict[agent_name] = scores
                        break
            
            if len(data_dict) == 3:  # All agents found
                # Omnibus variance test
                omnibus = compare_variances_all(data_dict, test='levene', alpha=0.05)
                
                # Pairwise variance comparisons
                pairwise = compare_variances_pairwise(data_dict, test='levene', alpha=0.05)
                
                # Record results
                for agent_name, scores in data_dict.items():
                    variance_results.append({
                        'Pool': pool_name.capitalize(),
                        'Noise_Level': noise,
                        'Agent': agent_name,
                        'Mean': np.mean(scores),
                        'Variance': np.var(scores, ddof=1),
                        'StdDev': np.std(scores, ddof=1),
                        'CV': np.std(scores, ddof=1) / np.mean(scores),
                        'N': len(scores),
                        'Levene_Statistic': omnibus['statistic'],
                        'Levene_P_Value': omnibus['p_value'],
                        'Variances_Equal': not omnibus['significant']
                    })
    
    # Save variance analysis
    variance_df = pd.DataFrame(variance_results)
    variance_df.to_csv(analysis_dir / 'variance_analysis_dbs_aifc_qlc.csv', index=False)
    print(f"    ✓ Saved to {analysis_dir / 'variance_analysis_dbs_aifc_qlc.csv'}")
    
    # Create pairwise comparison summary
    print(f"\n    Generating pairwise variance comparison summary...")
    pairwise_summary = []
    
    for pool_name in ['static', 'learning']:
        pool_dir = Path(f'results/main/{pool_name}')
        
        for noise in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
            data_dict = {}
            for agent_idx, agent_name in agent_indices.items():
                for d in pool_dir.iterdir():
                    if d.is_dir() and d.name.startswith(f"{agent_idx}_"):
                        scores = load_agent_repetition_scores(pool_dir, d.name, noise)
                        if len(scores) > 0:
                            data_dict[agent_name] = scores
                        break
            
            if len(data_dict) == 3:
                pairwise = compare_variances_pairwise(data_dict, test='levene', alpha=0.05)
                
                for _, row in pairwise.iterrows():
                    pairwise_summary.append({
                        'Pool': pool_name.capitalize(),
                        'Noise_Level': noise,
                        'Agent_1': row['group1'],
                        'Agent_2': row['group2'],
                        'Var_1': row['var1'],
                        'Var_2': row['var2'],
                        'Var_Ratio': row['var_ratio'],
                        'Levene_Statistic': row['test_statistic'],
                        'P_Value': row['p_value'],
                        'Significant': row['significant']
                    })
    
    pairwise_df = pd.DataFrame(pairwise_summary)
    pairwise_df.to_csv(analysis_dir / 'variance_pairwise_comparison.csv', index=False)
    print(f"    ✓ Saved to {analysis_dir / 'variance_pairwise_comparison.csv'}")


def visualize_variance_analysis(analysis_dir: Path, plots_dir: Path):
    """Create visualizations for variance analysis."""
    print("\n  Creating variance visualizations...")
    
    # Load the variance analysis data
    variance_df = pd.read_csv(analysis_dir / 'variance_analysis_dbs_aifc_qlc.csv')
    pairwise_df = pd.read_csv(analysis_dir / 'variance_pairwise_comparison.csv')
    
    # Create variance plots subdirectory
    variance_plots_dir = plots_dir / 'variance_analysis'
    variance_plots_dir.mkdir(exist_ok=True)
    
    # 1. Plot: Variance vs Noise Level for each agent
    print("    Creating variance vs noise plots...")
    for pool in ['Static', 'Learning']:
        pool_data = variance_df[variance_df['Pool'] == pool]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Variance
        for agent in pool_data['Agent'].unique():
            agent_data = pool_data[pool_data['Agent'] == agent]
            ax1.plot(agent_data['Noise_Level'], agent_data['Variance'], 
                    marker='o', label=agent, linewidth=2, markersize=6)
        ax1.set_xlabel('Noise Level', fontsize=11)
        ax1.set_ylabel('Variance', fontsize=11)
        ax1.set_title(f'{pool} Pool: Variance vs Noise Level', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Standard Deviation
        for agent in pool_data['Agent'].unique():
            agent_data = pool_data[pool_data['Agent'] == agent]
            ax2.plot(agent_data['Noise_Level'], agent_data['StdDev'], 
                    marker='o', label=agent, linewidth=2, markersize=6)
        ax2.set_xlabel('Noise Level', fontsize=11)
        ax2.set_ylabel('Standard Deviation', fontsize=11)
        ax2.set_title(f'{pool} Pool: StdDev vs Noise Level', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Coefficient of Variation
        for agent in pool_data['Agent'].unique():
            agent_data = pool_data[pool_data['Agent'] == agent]
            ax3.plot(agent_data['Noise_Level'], agent_data['CV'], 
                    marker='o', label=agent, linewidth=2, markersize=6)
        ax3.set_xlabel('Noise Level', fontsize=11)
        ax3.set_ylabel('Coefficient of Variation', fontsize=11)
        ax3.set_title(f'{pool} Pool: CV vs Noise Level', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Mean Score (for context)
        for agent in pool_data['Agent'].unique():
            agent_data = pool_data[pool_data['Agent'] == agent]
            ax4.plot(agent_data['Noise_Level'], agent_data['Mean'], 
                    marker='o', label=agent, linewidth=2, markersize=6)
        ax4.set_xlabel('Noise Level', fontsize=11)
        ax4.set_ylabel('Mean Score', fontsize=11)
        ax4.set_title(f'{pool} Pool: Mean Score vs Noise Level', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(variance_plots_dir / f'{pool.lower()}_variance_metrics.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Plot: Variance Ratio Heatmaps
    print("    Creating variance ratio heatmaps...")
    for pool in ['Static', 'Learning']:
        pool_pairwise = pairwise_df[pairwise_df['Pool'] == pool]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        noise_levels = sorted(pool_pairwise['Noise_Level'].unique())
        
        for idx, noise in enumerate(noise_levels):
            noise_data = pool_pairwise[pool_pairwise['Noise_Level'] == noise]
            
            # Create pivot table for heatmap
            agents = ['AIF-C', 'DBS', 'QL-C']
            matrix = np.zeros((3, 3))
            
            for i, agent1 in enumerate(agents):
                for j, agent2 in enumerate(agents):
                    if i == j:
                        matrix[i, j] = 1.0
                    else:
                        row = noise_data[
                            ((noise_data['Agent_1'] == agent1) & (noise_data['Agent_2'] == agent2)) |
                            ((noise_data['Agent_1'] == agent2) & (noise_data['Agent_2'] == agent1))
                        ]
                        if not row.empty:
                            ratio = row['Var_Ratio'].values[0]
                            if row['Agent_1'].values[0] == agent2:
                                ratio = 1 / ratio
                            matrix[i, j] = ratio
            
            # Plot heatmap
            im = axes[idx].imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0.5, vmax=2.0)
            axes[idx].set_xticks(range(3))
            axes[idx].set_yticks(range(3))
            axes[idx].set_xticklabels(agents)
            axes[idx].set_yticklabels(agents)
            axes[idx].set_title(f'Noise {noise:.2f}', fontsize=11)
            
            # Add text annotations
            for i in range(3):
                for j in range(3):
                    text = axes[idx].text(j, i, f'{matrix[i, j]:.2f}',
                                        ha="center", va="center", color="black", fontsize=9)
            
            # Add colorbar to last plot
            if idx == len(noise_levels) - 1:
                cbar = plt.colorbar(im, ax=axes[idx])
                cbar.set_label('Variance Ratio', rotation=270, labelpad=15)
        
        fig.suptitle(f'{pool} Pool: Variance Ratios Across Noise Levels\n(Row Agent Var / Column Agent Var)', 
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(variance_plots_dir / f'{pool.lower()}_variance_ratio_heatmap.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Plot: P-value significance across noise levels
    print("    Creating p-value significance plots...")
    for pool in ['Static', 'Learning']:
        pool_pairwise = pairwise_df[pairwise_df['Pool'] == pool]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot p-values for each agent pair
        for agent_pair in pool_pairwise[['Agent_1', 'Agent_2']].drop_duplicates().values:
            pair_label = f"{agent_pair[0]} vs {agent_pair[1]}"
            pair_data = pool_pairwise[
                ((pool_pairwise['Agent_1'] == agent_pair[0]) & (pool_pairwise['Agent_2'] == agent_pair[1])) |
                ((pool_pairwise['Agent_1'] == agent_pair[1]) & (pool_pairwise['Agent_2'] == agent_pair[0]))
            ].sort_values('Noise_Level')
            
            ax.plot(pair_data['Noise_Level'], pair_data['P_Value'], 
                   marker='o', label=pair_label, linewidth=2, markersize=6)
        
        ax.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
        ax.set_xlabel('Noise Level', fontsize=11)
        ax.set_ylabel('P-value (Levene Test)', fontsize=11)
        ax.set_title(f'{pool} Pool: Variance Equality Test P-values', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(variance_plots_dir / f'{pool.lower()}_pvalue_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Plot: Bar chart comparing variances at specific noise levels
    print("    Creating variance comparison bar charts...")
    comparison_noises = [0.0, 0.15, 0.25]
    
    for pool in ['Static', 'Learning']:
        pool_data = variance_df[variance_df['Pool'] == pool]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, noise in enumerate(comparison_noises):
            noise_data = pool_data[pool_data['Noise_Level'] == noise]
            
            agents = noise_data['Agent'].values
            variances = noise_data['Variance'].values
            
            bars = axes[idx].bar(agents, variances, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            axes[idx].set_ylabel('Variance', fontsize=11)
            axes[idx].set_title(f'Noise Level {noise:.2f}', fontsize=11, fontweight='bold')
            axes[idx].grid(True, axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.0f}',
                             ha='center', va='bottom', fontsize=10)
        
        fig.suptitle(f'{pool} Pool: Variance Comparison at Key Noise Levels', 
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(variance_plots_dir / f'{pool.lower()}_variance_bars.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"    ✓ Variance visualizations saved to {variance_plots_dir}")


def calculate_cvar(values: np.ndarray, alpha: float = 0.05) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
    
    CVaR is the average of the worst alpha% of outcomes.
    
    Args:
        values: Array of values (scores, rates, etc.)
        alpha: Confidence level (e.g., 0.05 for worst 5%)
        
    Returns:
        CVaR value (average of worst alpha% outcomes)
    """
    sorted_values = np.sort(values)
    cutoff_index = max(1, int(np.ceil(alpha * len(sorted_values))))
    return np.mean(sorted_values[:cutoff_index])


def calculate_metric_cvar(pool_dir: Path, agent_dir: str, noise_level: float, 
                         metric: str = 'score', alpha: float = 0.05) -> float:
    """
    Calculate CVaR for a specific metric (score, cc_rate, cd_rate, or norm_coop).
    
    Args:
        pool_dir: Path to pool directory
        agent_dir: Agent directory name
        noise_level: Noise level
        metric: 'score', 'cc_rate', 'cd_rate', or 'norm_coop'
        alpha: Confidence level
        
    Returns:
        CVaR value for the metric
    """
    from experiments.result_utils import load_tournament_results
    
    df = load_tournament_results(pool_dir, agent_dir, noise_level)
    if df.empty:
        return np.nan
    
    agent_data = df[df['Player index'] == 0]
    
    # Calculate metric per repetition
    values = []
    for rep in agent_data['repetition'].unique():
        rep_data = agent_data[agent_data['repetition'] == rep]
        
        if metric == 'score':
            values.append(rep_data['Score'].mean())
        elif metric == 'cc_rate':
            total_turns = rep_data['Turns'].sum()
            cc_count = rep_data['CC count'].sum()
            if total_turns > 0:
                values.append(cc_count / total_turns)
        elif metric == 'cd_rate':
            total_turns = rep_data['Turns'].sum()
            cd_count = rep_data['CD count'].sum()
            if total_turns > 0:
                values.append(cd_count / total_turns)
        elif metric == 'norm_coop':
            cc_count = rep_data['CC count'].sum()
            cd_count = rep_data['CD count'].sum()
            total_coop = cc_count + cd_count
            if total_coop > 0:
                values.append(cc_count / total_coop)
    
    if values:
        return calculate_cvar(np.array(values), alpha)
    return np.nan


def generate_cvar_analysis(plots_dir: Path, rename_map: dict):
    """Generate CVaR analysis comparing architectures (AIF vs QL vs BQL) within strategy types."""
    print("\n  Generating CVaR (5%) analysis comparing architectures...")
    
    # Agent groups: Compare architectures within strategy types
    architecture_groups = {
        'cooperative': {
            'AIF-C': '1',
            'AIF-C-N': '7',
            'QL-C': '3',
            'BQL-C': '5',
        },
        'rational': {
            'AIF-R': '0',
            'QL-R': '2',
            'BQL-R': '4',
            'AIF-R-N': '6',
        }
    }
    
    # Create CVaR plots subdirectory
    cvar_plots_dir = plots_dir / 'cvar_analysis'
    cvar_plots_dir.mkdir(exist_ok=True)
    
    noise_levels_list = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    
    for pool_name in ['static', 'learning']:
        pool_dir = Path(f'results/main/{pool_name}')
        plots_subdir = cvar_plots_dir / pool_name
        plots_subdir.mkdir(exist_ok=True)
        
        print(f"    Processing {pool_name} pool...")
        
        # Generate CVaR plots for each strategy type (cooperative vs rational)
        for strategy_type, agents in architecture_groups.items():
            print(f"      Generating {strategy_type} architecture comparison CVaR plots...")
            
            # Collect data for all architectures in this strategy type
            cvar_data = {
                'scores': {},
                'cc_rates': {},
                'cd_rates': {},
                'norm_coop': {}
            }
            
            for agent_name, agent_idx in agents.items():
                # Find the actual directory
                agent_dir = None
                for d in pool_dir.iterdir():
                    if d.is_dir() and d.name.startswith(f"{agent_idx}_"):
                        agent_dir = d.name
                        break
                
                if agent_dir:
                    score_cvars = []
                    cc_cvars = []
                    cd_cvars = []
                    norm_coop_cvars = []
                    
                    for noise in noise_levels_list:
                        score_cvar = calculate_metric_cvar(pool_dir, agent_dir, noise, 'score')
                        cc_cvar = calculate_metric_cvar(pool_dir, agent_dir, noise, 'cc_rate')
                        cd_cvar = calculate_metric_cvar(pool_dir, agent_dir, noise, 'cd_rate')
                        norm_coop_cvar = calculate_metric_cvar(pool_dir, agent_dir, noise, 'norm_coop')
                        
                        score_cvars.append(score_cvar)
                        cc_cvars.append(cc_cvar)
                        cd_cvars.append(cd_cvar)
                        norm_coop_cvars.append(norm_coop_cvar)
                    
                    cvar_data['scores'][agent_name] = score_cvars
                    cvar_data['cc_rates'][agent_name] = cc_cvars
                    cvar_data['cd_rates'][agent_name] = cd_cvars
                    cvar_data['norm_coop'][agent_name] = norm_coop_cvars
            
            # Plot CVaR for Scores
            fig, ax = plt.subplots(figsize=(10, 6))
            for agent_name, values in cvar_data['scores'].items():
                ax.plot(noise_levels_list, values, marker='o', 
                       label=agent_name, linewidth=2, markersize=6)
            ax.set_xlabel('Noise Level', fontsize=11)
            ax.set_ylabel('CVaR Score (Worst 5%)', fontsize=11)
            ax.set_title(f'{pool_name.capitalize()} Pool: {strategy_type.capitalize()} Architectures - CVaR Score vs Noise', 
                        fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_subdir / f'{pool_name}_{strategy_type}_cvar_scores.pdf', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot CVaR for CC Rates
            fig, ax = plt.subplots(figsize=(10, 6))
            for agent_name, values in cvar_data['cc_rates'].items():
                ax.plot(noise_levels_list, values, marker='o', 
                       label=agent_name, linewidth=2, markersize=6)
            ax.set_xlabel('Noise Level', fontsize=11)
            ax.set_ylabel('CVaR CC Rate (Worst 5%)', fontsize=11)
            ax.set_title(f'{pool_name.capitalize()} Pool: {strategy_type.capitalize()} Architectures - CVaR CC Rate vs Noise', 
                        fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
            plt.tight_layout()
            plt.savefig(plots_subdir / f'{pool_name}_{strategy_type}_cvar_cc_rate.pdf', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot CVaR for Normalized Cooperation
            fig, ax = plt.subplots(figsize=(10, 6))
            for agent_name, values in cvar_data['norm_coop'].items():
                ax.plot(noise_levels_list, values, marker='o', 
                       label=agent_name, linewidth=2, markersize=6)
            ax.set_xlabel('Noise Level', fontsize=11)
            ax.set_ylabel('CVaR Normalized Cooperation (Worst 5%)', fontsize=11)
            ax.set_title(f'{pool_name.capitalize()} Pool: {strategy_type.capitalize()} Architectures - CVaR Normalized Cooperation vs Noise', 
                        fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
            plt.tight_layout()
            plt.savefig(plots_subdir / f'{pool_name}_{strategy_type}_cvar_norm_coop.pdf', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot CVaR for CD Rates
            fig, ax = plt.subplots(figsize=(10, 6))
            for agent_name, values in cvar_data['cd_rates'].items():
                ax.plot(noise_levels_list, values, marker='o', 
                       label=agent_name, linewidth=2, markersize=6)
            ax.set_xlabel('Noise Level', fontsize=11)
            ax.set_ylabel('CVaR CD Rate (Worst 5%)', fontsize=11)
            ax.set_title(f'{pool_name.capitalize()} Pool: {strategy_type.capitalize()} Architectures - CVaR CD Rate vs Noise', 
                        fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 0.45)
            plt.tight_layout()
            plt.savefig(plots_subdir / f'{pool_name}_{strategy_type}_cvar_cd_rate.pdf', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"      ✓ Saved {pool_name} CVaR plots")
    
    print(f"    ✓ CVaR visualizations saved to {cvar_plots_dir}")


def generate_comprehensive_analysis_csv(analysis_dir: Path, rename_map: dict):
    """Generate a comprehensive CSV with all analysis data per agent per noise level."""
    print("\n" + "="*60)
    print("Generating comprehensive analysis CSV...")
    print("="*60)
    
    from experiments.result_utils import load_tournament_results, get_agent_dirs
    from experiments.stats_utils import compare_variances_pairwise, compare_variances_all
    
    comprehensive_data = []
    
    # Reference agents for variance comparisons: DBS (8), AIF-C (1), QL-C (3)
    reference_agents = {
        '8': 'DBS',
        '1': 'AIF-C',
        '3': 'QL-C'
    }
    
    noise_levels_list = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    
    for pool_name in ['static', 'learning']:
        print(f"\n  Processing {pool_name} pool...")
        pool_dir = Path(f'results/main/{pool_name}')
        agent_dirs = get_agent_dirs(pool_dir)
        
        # Load reference agent scores for variance comparisons
        reference_scores = {}
        for agent_idx, agent_name in reference_agents.items():
            # Find the actual directory
            agent_dir = None
            for d in pool_dir.iterdir():
                if d.is_dir() and d.name.startswith(f"{agent_idx}_"):
                    agent_dir = d.name
                    break
            
            if agent_dir:
                reference_scores[agent_idx] = {}
                for noise in noise_levels_list:
                    scores = load_agent_repetition_scores(pool_dir, agent_dir, noise)
                    if len(scores) > 0:
                        reference_scores[agent_idx][noise] = scores
        
        # Process each agent
        for agent_dir_name, display_name in agent_dirs:
            agent_idx = agent_dir_name.split('_')[0]
            agent_display_name = rename_map.get(agent_idx, display_name)
            
            print(f"    Processing {agent_display_name} ({agent_dir_name})...")
            
            for noise in noise_levels_list:
                # Load tournament results
                df = load_tournament_results(pool_dir, agent_dir_name, noise)
                if df.empty:
                    continue
                
                agent_data = df[df['Player index'] == 0]
                
                # Calculate per-repetition metrics
                score_values = []
                cc_rate_values = []
                cd_rate_values = []
                norm_coop_values = []
                
                for rep in agent_data['repetition'].unique():
                    rep_data = agent_data[agent_data['repetition'] == rep]
                    
                    # Score
                    score_values.append(rep_data['Score'].mean())
                    
                    # CC rate
                    total_turns = rep_data['Turns'].sum()
                    cc_count = rep_data['CC count'].sum()
                    if total_turns > 0:
                        cc_rate_values.append(cc_count / total_turns)
                    
                    # CD rate
                    cd_count = rep_data['CD count'].sum()
                    if total_turns > 0:
                        cd_rate_values.append(cd_count / total_turns)
                    
                    # Normalized cooperation
                    total_coop = cc_count + cd_count
                    if total_coop > 0:
                        norm_coop_values.append(cc_count / total_coop)
                
                if not score_values:
                    continue
                
                score_arr = np.array(score_values)
                cc_arr = np.array(cc_rate_values) if cc_rate_values else np.array([])
                cd_arr = np.array(cd_rate_values) if cd_rate_values else np.array([])
                norm_coop_arr = np.array(norm_coop_values) if norm_coop_values else np.array([])
                
                n = len(score_arr)
                
                # Score statistics
                score_mean = np.mean(score_arr)
                score_std = np.std(score_arr, ddof=1)
                score_var = np.var(score_arr, ddof=1)
                score_cv = score_std / score_mean if score_mean != 0 else np.nan
                score_min = np.min(score_arr)
                score_max = np.max(score_arr)
                score_median = np.median(score_arr)
                score_q25 = np.percentile(score_arr, 25)
                score_q75 = np.percentile(score_arr, 75)
                score_iqr = score_q75 - score_q25
                score_cvar = calculate_cvar(score_arr, alpha=0.05)
                
                # 95% CI for scores
                if n > 1:
                    ci = stats.t.interval(0.95, n-1, loc=score_mean, scale=score_std/np.sqrt(n))
                    score_ci_lower = ci[0]
                    score_ci_upper = ci[1]
                else:
                    score_ci_lower = score_mean
                    score_ci_upper = score_mean
                
                # CC rate statistics
                if len(cc_arr) > 0:
                    cc_mean = np.mean(cc_arr)
                    cc_std = np.std(cc_arr, ddof=1)
                    cc_var = np.var(cc_arr, ddof=1)
                    cc_cv = cc_std / cc_mean if cc_mean != 0 else np.nan
                    cc_min = np.min(cc_arr)
                    cc_max = np.max(cc_arr)
                    cc_median = np.median(cc_arr)
                    cc_q25 = np.percentile(cc_arr, 25)
                    cc_q75 = np.percentile(cc_arr, 75)
                    cc_iqr = cc_q75 - cc_q25
                    cc_cvar = calculate_cvar(cc_arr, alpha=0.05)
                    
                    if n > 1:
                        ci = stats.t.interval(0.95, n-1, loc=cc_mean, scale=cc_std/np.sqrt(n))
                        cc_ci_lower = max(0, ci[0])
                        cc_ci_upper = min(1, ci[1])
                    else:
                        cc_ci_lower = cc_mean
                        cc_ci_upper = cc_mean
                else:
                    cc_mean = np.nan
                    cc_std = np.nan
                    cc_var = np.nan
                    cc_cv = np.nan
                    cc_min = np.nan
                    cc_max = np.nan
                    cc_median = np.nan
                    cc_q25 = np.nan
                    cc_q75 = np.nan
                    cc_iqr = np.nan
                    cc_cvar = np.nan
                    cc_ci_lower = np.nan
                    cc_ci_upper = np.nan
                
                # CD rate statistics
                if len(cd_arr) > 0:
                    cd_mean = np.mean(cd_arr)
                    cd_std = np.std(cd_arr, ddof=1)
                    cd_var = np.var(cd_arr, ddof=1)
                    cd_cv = cd_std / cd_mean if cd_mean != 0 else np.nan
                    cd_min = np.min(cd_arr)
                    cd_max = np.max(cd_arr)
                    cd_median = np.median(cd_arr)
                    cd_q25 = np.percentile(cd_arr, 25)
                    cd_q75 = np.percentile(cd_arr, 75)
                    cd_iqr = cd_q75 - cd_q25
                    cd_cvar = calculate_cvar(cd_arr, alpha=0.05)
                    
                    if n > 1:
                        ci = stats.t.interval(0.95, n-1, loc=cd_mean, scale=cd_std/np.sqrt(n))
                        cd_ci_lower = max(0, ci[0])
                        cd_ci_upper = min(1, ci[1])
                    else:
                        cd_ci_lower = cd_mean
                        cd_ci_upper = cd_mean
                else:
                    cd_mean = np.nan
                    cd_std = np.nan
                    cd_var = np.nan
                    cd_cv = np.nan
                    cd_min = np.nan
                    cd_max = np.nan
                    cd_median = np.nan
                    cd_q25 = np.nan
                    cd_q75 = np.nan
                    cd_iqr = np.nan
                    cd_cvar = np.nan
                    cd_ci_lower = np.nan
                    cd_ci_upper = np.nan
                
                # Normalized cooperation statistics
                if len(norm_coop_arr) > 0:
                    norm_coop_mean = np.mean(norm_coop_arr)
                    norm_coop_std = np.std(norm_coop_arr, ddof=1)
                    norm_coop_var = np.var(norm_coop_arr, ddof=1)
                    norm_coop_cv = norm_coop_std / norm_coop_mean if norm_coop_mean != 0 else np.nan
                    norm_coop_min = np.min(norm_coop_arr)
                    norm_coop_max = np.max(norm_coop_arr)
                    norm_coop_median = np.median(norm_coop_arr)
                    norm_coop_q25 = np.percentile(norm_coop_arr, 25)
                    norm_coop_q75 = np.percentile(norm_coop_arr, 75)
                    norm_coop_iqr = norm_coop_q75 - norm_coop_q25
                    norm_coop_cvar = calculate_cvar(norm_coop_arr, alpha=0.05)
                    
                    if len(norm_coop_arr) > 1:
                        ci = stats.t.interval(0.95, len(norm_coop_arr)-1, loc=norm_coop_mean, scale=norm_coop_std/np.sqrt(len(norm_coop_arr)))
                        norm_coop_ci_lower = max(0, ci[0])
                        norm_coop_ci_upper = min(1, ci[1])
                    else:
                        norm_coop_ci_lower = norm_coop_mean
                        norm_coop_ci_upper = norm_coop_mean
                else:
                    norm_coop_mean = np.nan
                    norm_coop_std = np.nan
                    norm_coop_var = np.nan
                    norm_coop_cv = np.nan
                    norm_coop_min = np.nan
                    norm_coop_max = np.nan
                    norm_coop_median = np.nan
                    norm_coop_q25 = np.nan
                    norm_coop_q75 = np.nan
                    norm_coop_iqr = np.nan
                    norm_coop_cvar = np.nan
                    norm_coop_ci_lower = np.nan
                    norm_coop_ci_upper = np.nan
                
                # Variance comparisons with reference agents
                # Initialize all comparison columns to NaN
                variance_comparisons = {}
                for ref_idx, ref_name in reference_agents.items():
                    variance_comparisons[f'vs_{ref_name}_var_ratio'] = np.nan
                    variance_comparisons[f'vs_{ref_name}_levene_stat'] = np.nan
                    variance_comparisons[f'vs_{ref_name}_levene_p'] = np.nan
                    variance_comparisons[f'vs_{ref_name}_levene_sig'] = np.nan
                    variance_comparisons[f'vs_{ref_name}_f_stat'] = np.nan
                    variance_comparisons[f'vs_{ref_name}_f_p'] = np.nan
                
                # Perform comparisons for each reference agent
                for ref_idx, ref_name in reference_agents.items():
                    # Skip comparison if this agent is the reference agent
                    if agent_idx == ref_idx:
                        continue
                    
                    if ref_idx in reference_scores and noise in reference_scores[ref_idx]:
                        ref_scores = reference_scores[ref_idx][noise]
                        
                        # Perform Levene's test for variance equality
                        try:
                            levene_stat, levene_p = stats.levene(score_arr, ref_scores)
                            
                            # F-test for variance (ratio of variances)
                            ref_var = np.var(ref_scores, ddof=1)
                            if ref_var > 0:
                                f_stat = score_var / ref_var if score_var >= ref_var else ref_var / score_var
                                df1 = len(score_arr) - 1
                                df2 = len(ref_scores) - 1
                                if score_var >= ref_var:
                                    f_p = 2 * (1 - stats.f.cdf(f_stat, df1, df2))
                                else:
                                    f_p = 2 * (1 - stats.f.cdf(f_stat, df2, df1))
                            else:
                                f_stat = np.nan
                                f_p = np.nan
                            
                            var_ratio = score_var / ref_var if ref_var > 0 else np.nan
                            
                            variance_comparisons[f'vs_{ref_name}_var_ratio'] = var_ratio
                            variance_comparisons[f'vs_{ref_name}_levene_stat'] = levene_stat
                            variance_comparisons[f'vs_{ref_name}_levene_p'] = levene_p
                            variance_comparisons[f'vs_{ref_name}_levene_sig'] = levene_p < 0.05
                            variance_comparisons[f'vs_{ref_name}_f_stat'] = f_stat
                            variance_comparisons[f'vs_{ref_name}_f_p'] = f_p
                        except Exception as e:
                            # If comparison fails, leave as NaN (already initialized)
                            pass
                
                # Create row entry
                row = {
                    'pool': pool_name.capitalize(),
                    'agent_index': agent_idx,
                    'agent_name': agent_display_name,
                    'agent_dir': agent_dir_name,
                    'noise_level': noise,
                    'n_repetitions': n,
                    
                    # Score metrics
                    'score_mean': score_mean,
                    'score_std': score_std,
                    'score_variance': score_var,
                    'score_cv': score_cv,
                    'score_min': score_min,
                    'score_max': score_max,
                    'score_median': score_median,
                    'score_q25': score_q25,
                    'score_q75': score_q75,
                    'score_iqr': score_iqr,
                    'score_ci_lower': score_ci_lower,
                    'score_ci_upper': score_ci_upper,
                    'score_cvar_5pct': score_cvar,
                    
                    # CC rate metrics
                    'cc_rate_mean': cc_mean,
                    'cc_rate_std': cc_std,
                    'cc_rate_variance': cc_var,
                    'cc_rate_cv': cc_cv,
                    'cc_rate_min': cc_min,
                    'cc_rate_max': cc_max,
                    'cc_rate_median': cc_median,
                    'cc_rate_q25': cc_q25,
                    'cc_rate_q75': cc_q75,
                    'cc_rate_iqr': cc_iqr,
                    'cc_rate_ci_lower': cc_ci_lower,
                    'cc_rate_ci_upper': cc_ci_upper,
                    'cc_rate_cvar_5pct': cc_cvar,
                    
                    # CD rate metrics
                    'cd_rate_mean': cd_mean,
                    'cd_rate_std': cd_std,
                    'cd_rate_variance': cd_var,
                    'cd_rate_cv': cd_cv,
                    'cd_rate_min': cd_min,
                    'cd_rate_max': cd_max,
                    'cd_rate_median': cd_median,
                    'cd_rate_q25': cd_q25,
                    'cd_rate_q75': cd_q75,
                    'cd_rate_iqr': cd_iqr,
                    'cd_rate_ci_lower': cd_ci_lower,
                    'cd_rate_ci_upper': cd_ci_upper,
                    'cd_rate_cvar_5pct': cd_cvar,
                    
                    # Normalized cooperation metrics
                    'norm_coop_mean': norm_coop_mean,
                    'norm_coop_std': norm_coop_std,
                    'norm_coop_variance': norm_coop_var,
                    'norm_coop_cv': norm_coop_cv,
                    'norm_coop_min': norm_coop_min,
                    'norm_coop_max': norm_coop_max,
                    'norm_coop_median': norm_coop_median,
                    'norm_coop_q25': norm_coop_q25,
                    'norm_coop_q75': norm_coop_q75,
                    'norm_coop_iqr': norm_coop_iqr,
                    'norm_coop_ci_lower': norm_coop_ci_lower,
                    'norm_coop_ci_upper': norm_coop_ci_upper,
                    'norm_coop_cvar_5pct': norm_coop_cvar,
                }
                
                # Add variance comparisons
                row.update(variance_comparisons)
                
                comprehensive_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(comprehensive_data)
    
    # Sort by pool, agent_index, noise_level
    df = df.sort_values(['pool', 'agent_index', 'noise_level']).reset_index(drop=True)
    
    # Save to CSV
    output_path = analysis_dir / 'comprehensive_analysis.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n  ✓ Comprehensive analysis CSV saved to {output_path}")
    print(f"  ✓ Total rows: {len(df)}")
    print(f"  ✓ Columns: {len(df.columns)}")
    
    return df


if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    # Run experiments
    run_experiment(strategies, static_pool, "static")
    run_experiment(strategies, learn_pool, "learning")
    
    # Generate plots
    generate_plots()