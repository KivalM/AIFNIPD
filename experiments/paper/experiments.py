from typing import Any
from tournament.handler import FileSystemHandler
from tournament.tournament import NoiseTournament
from experiments import static_pool, learn_pool
import itertools
import multiprocessing
from agents.aif.jax.aif import ActiveInferenceAgent
from agents.aif.jax.epsilon_greedy import EpsilonGreedyAIFAgent
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
noise_levels = list(np.arange(0, 0.10, 0.05).round(2))
repetitions = 30
turns = 1000
processes = 30
seed = 42

# Define games
games = {
    "pd": axl.Game(r=3, s=0, t=5, p=1),
    "chicken": axl.Game(r=3, s=1, t=4, p=0),
    "stag": axl.Game(r=4, s=0, t=3, p=3),
}

strategies = [
    # QLearner_0.9_0.9_0.1_1_0.999
    JaxQLearner(
        learning_rate=0.9,
        discount_rate=0.9,
        action_selection_parameter=0.1,
        memory_length=1,
        decay_rate=0.999,
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
    # Epistemic Active Inference
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
    # Utility Active Inference
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
    # Noisy Epistemic Active Inference
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
        use_noisy_observation_model=True,
    ),
    # Noisy Utility Active Inference
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
        use_noisy_observation_model=True,
        use_states_info_gain=False,
        use_param_info_gain=False,
    ),
    # Epsilon-Greedy Active Inference
    EpsilonGreedyAIFAgent(
        pB_scale=1,
        gamma=1,
        alpha=1,
        bias=0.5,
        cooperative_preference=False,
        policy_len=5,
        update_interval=10,
        seed=seed,
        lr_B=1,
        epsilon_start=0.5,
        epsilon_decay=0.99,
        epsilon_min=0.01,
    ),
]

def run_experiment(strategies, opponents, dir_name):
    for game_name, game in games.items():
        print(f"\nRunning experiments for {game_name}...")
        for i, strategy in enumerate(strategies):
            handler = FileSystemHandler(root_dir=f"results/main/{game_name}/{dir_name}/{i}_{strategy.__class__.__name__}")
            noise_tournament = NoiseTournament(
                players=[strategy] + opponents,
                noise_levels=noise_levels,
                repetitions=repetitions,
                seed=seed,
                game=game,
                callback=handler.save_results,
                skip_callback=handler.skip_run,
            )
            noise_tournament.run(turns=turns, processes=processes)
            print(f"Results for {game_name} are saved in the '{handler.root_dir}' directory.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    # Run experiments
    run_experiment(strategies, static_pool, "static")
    run_experiment(strategies, learn_pool, "learning")
    