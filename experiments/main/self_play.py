from tournament.handler import FileSystemHandler
from tournament.tournament import NoiseTournament
import multiprocessing
from agents.aif.jax.aif import ActiveInferenceAgent
from agents.bqlearner import JaxBayesianQLearner, CooperativeBQLearner
from agents.qlearner import JaxQLearner, CooperativeQLearner
from agents.psrl import PSRL, CooperativePSRL
from agents.dynaQ import DynaQ, CooperativeDynaQ
import axelrod as axl
import numpy as np
from pathlib import Path
from experiments import (
    generate_selfplay_heatmaps,
    create_combined_heatmap_grid,
    export_matrices_to_csv,
    generate_selfplay_comparison_tables,
    generate_selfplay_diagonal_comparison_tables,
    setup_publication_style,
)

# Environment parameters
noise_levels = list(np.arange(0, 0.30, 0.05).round(2))
repetitions = 5
turns = 1000
processes = 10
seed = 42

strategies = [
    JaxQLearner(
        learning_rate=0.9,
        discount_rate=0.9,
        action_selection_parameter=0.1,
    ),

    JaxBayesianQLearner(
        discount_rate=0.5,
        initial_variance=1.0,
        reward_variance=1.0,
    ),
    axl.DBS(0.999, 5, 2, 3, 5),
    axl.GTFT(),
    axl.ContriteTitForTat(),
    DynaQ(
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
    ActiveInferenceAgent(
        pB_scale=1,
        gamma=1,
        alpha=1,
        bias=0.5,
        cooperative_preference=False,
        policy_len=5,
        update_interval=10,
        seed=seed,
        lr_B=1,
        action_selection="deterministic",
    ),
    ActiveInferenceAgent(
        pB_scale=1,
        gamma=1,
        alpha=1,
        bias=0.5,
        cooperative_preference=False,
        policy_len=5,
        update_interval=10,
        seed=seed,
        lr_B=1,
        action_selection="deterministic",
        use_states_info_gain=False,
        use_param_info_gain=False,
    ),
    ActiveInferenceAgent(
        pB_scale=1,
        gamma=1,
        alpha=1,
        bias=0.5,
        cooperative_preference=False,
        policy_len=5,
        update_interval=10,
        seed=seed,
        lr_B=1,
        action_selection="deterministic",
    ),
]

def run_self_play_experiment(strategies):
    """Run a single round-robin tournament with all strategies."""
    handler = FileSystemHandler(root_dir="results/sp")
    noise_tournament = NoiseTournament(
        players=strategies,  # All strategies play against each other in round-robin
        noise_levels=noise_levels,
        repetitions=repetitions,
        seed=seed,
        callback=handler.save_results,
        skip_callback=handler.skip_run,
    )
    noise_tournament.run(turns=turns, processes=processes)
    print(f"Results are saved in the '{handler.root_dir}' directory.")


def analyze_results():
    """Analyze and visualize self-play tournament results."""
    print("\n" + "="*60)
    print("Analyzing Self-Play Tournament Results")
    print("="*60)
    
    # Setup
    sp_dir = Path("results/sp")
    output_dir = Path("results/sp/heatmaps")
    csv_output_dir = Path("results/sp/matrices")
    
    # Agent rename map for cleaner labels
    rename_map = {
        0: 'QL',
        1: 'BQL',
        2: 'DBS',
        3: 'GTFT',
        4: 'CTFT',
        5: 'DynaQ',
        6: 'PSRL',
        7: 'AIF-S',
        8: 'AIF-S-U',
        9: 'AIF-S-N',
    }
    
    # Setup publication style
    setup_publication_style(use_latex=False)
    
    # Generate individual heatmaps for each noise level
    print("\n1. Generating individual heatmaps for each noise level...")
    generate_selfplay_heatmaps(
        sp_dir=sp_dir,
        noise_levels=noise_levels,
        output_dir=output_dir,
        rename_map=rename_map,
        figsize=(14, 12)
    )
    
    # Generate combined grid plots
    print("\n2. Generating combined grid plots...")
    
    print("  Creating CC rate grid...")
    create_combined_heatmap_grid(
        sp_dir=sp_dir,
        noise_levels=noise_levels,
        output_dir=output_dir,
        metric='cc_rate',
        rename_map=rename_map,
        max_cols=3
    )
    
    print("  Creating CD rate grid...")
    create_combined_heatmap_grid(
        sp_dir=sp_dir,
        noise_levels=noise_levels,
        output_dir=output_dir,
        metric='cd_rate',
        rename_map=rename_map,
        max_cols=3
    )
    
    print("  Creating normalized cooperation grid...")
    create_combined_heatmap_grid(
        sp_dir=sp_dir,
        noise_levels=noise_levels,
        output_dir=output_dir,
        metric='norm_coop',
        rename_map=rename_map,
        max_cols=3
    )
    
    print("  Creating score grid...")
    create_combined_heatmap_grid(
        sp_dir=sp_dir,
        noise_levels=noise_levels,
        output_dir=output_dir,
        metric='score',
        rename_map=rename_map,
        max_cols=3
    )
    
    # Export matrices to CSV for further analysis
    print("\n3. Exporting matrices to CSV...")
    export_matrices_to_csv(
        sp_dir=sp_dir,
        noise_levels=noise_levels,
        output_dir=csv_output_dir,
        rename_map=rename_map
    )
    
    # Generate comparison tables
    print("\n4. Generating comparison tables...")
    generate_selfplay_comparison_tables(
        sp_dir=sp_dir,
        output_dir=csv_output_dir,
        rename_map=rename_map,
        noise_levels=[0.0, 0.05]
    )
    
    # Generate diagonal (self vs self) comparison tables
    print("\n5. Generating diagonal (self-play) comparison tables...")
    generate_selfplay_diagonal_comparison_tables(
        sp_dir=sp_dir,
        output_dir=csv_output_dir,
        rename_map=rename_map,
        noise_levels=[0.0, 0.05]
    )
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Heatmaps saved to: {output_dir}")
    print(f"CSV matrices saved to: {csv_output_dir}")
    print(f"Comparison tables saved to: {csv_output_dir}")
    print("="*60)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    # Run self-play experiments
    run_self_play_experiment(strategies)
    
    # Analyze results
    analyze_results()

