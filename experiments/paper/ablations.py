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
noise_levels = list(np.arange(0, 0.55, 0.05).round(2))
repetitions = 30
turns = 1000
processes = 30
seed = 42

# Define games
games = {
    "pd": axl.Game(r=3, s=0, t=5, p=1),
}

strategies = [

]

def run_experiment_1():
    # POMDP formulation self play
    agent = ActiveInferenceAgent(
        lr_B=1,
        cooperative_preference=False,
        policy_len=5,
        update_interval=50,
        pB_scale=1,
        gamma=1,
        alpha=1,
        bias=0.5,
        use_noisy_observation_model=True,
        use_states_info_gain=False,
        use_param_info_gain=False,
    )
    handler = FileSystemHandler(root_dir=f"results/paper/ablations/1/")
    noise_tournament = NoiseTournament(
        players=[agent],
        noise_levels=noise_levels,
        repetitions=repetitions,
        seed=seed,
        game=games["pd"],
        callback=handler.save_results,
        skip_callback=handler.skip_run,
    )
    noise_tournament.run(turns=turns, processes=processes)
    print(f"Results are saved in the '{handler.root_dir}' directory.")

def run_experiment_2():
    # MDP formulation self play
    agent = ActiveInferenceAgent(
        lr_B=1,
        cooperative_preference=False,
        policy_len=3,
        update_interval=50,
        pB_scale=1,
        gamma=1,
        alpha=1,
        bias=0.5,
        use_noisy_observation_model=False,
        use_states_info_gain=False,
        use_param_info_gain=False,
    )
    handler = FileSystemHandler(root_dir=f"results/paper/ablations/2/")
    noise_tournament = NoiseTournament(
        players=[agent],
        noise_levels=noise_levels,
        repetitions=repetitions,
        seed=seed,
        game=games["pd"],
        callback=handler.save_results,
        skip_callback=handler.skip_run,
    )
    noise_tournament.run(turns=turns, processes=processes)
    print(f"Results are saved in the '{handler.root_dir}' directory.")

def run_experiment_3():
    # POMDP formulation X play
    agent = ActiveInferenceAgent(
        lr_B=1,
        cooperative_preference=False,
        policy_len=3,
        update_interval=10,
        pB_scale=1,
        gamma=1,
        alpha=1,
        bias=0.5,
        use_noisy_observation_model=True,
                use_states_info_gain=False,
        use_param_info_gain=False,
    )
    handler = FileSystemHandler(root_dir=f"results/paper/ablations/3/")
    noise_tournament = NoiseTournament(
        players=[agent, axl.TitForTat(), axl.WinStayLoseShift()],
        noise_levels=noise_levels,
        repetitions=repetitions,
        seed=seed,
        game=games["pd"],
        callback=handler.save_results,
        skip_callback=handler.skip_run,
    )
    noise_tournament.run(turns=turns, processes=processes)
    print(f"Results are saved in the '{handler.root_dir}' directory.")


def run_experiment_4():
    # MDP formulation X play
    agent = ActiveInferenceAgent(
        lr_B=1,
        cooperative_preference=False,
        policy_len=3,
        update_interval=5,
        pB_scale=1,
        gamma=1,
        alpha=1,
        bias=0.5,
        use_noisy_observation_model=False,
        use_states_info_gain=False,
        use_param_info_gain=False,
    )
    handler = FileSystemHandler(root_dir=f"results/paper/ablations/4/")
    noise_tournament = NoiseTournament(
        players=[agent, axl.TitForTat(), axl.WinStayLoseShift()],
        noise_levels=noise_levels,
        repetitions=repetitions,
        seed=seed,
        game=games["pd"],
        callback=handler.save_results,
        skip_callback=handler.skip_run,
    )
    noise_tournament.run(turns=turns, processes=processes)
    print(f"Results are saved in the '{handler.root_dir}' directory.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    
