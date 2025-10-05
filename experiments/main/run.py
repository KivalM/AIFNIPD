from tournament.handler import FileSystemHandler
from tournament.tournament import NoiseTournament
from experiments import static_pool, learn_pool
import itertools
import multiprocessing
from agents.aif.jax.five_state import JaxFiveStateAgent
from agents.aif.jax.five_state_noise import JaxFiveStateAgentNoisy
from agents.aif.jax.five_state_utility import JaxFiveStateAgentUtility
from agents.bqlearner import JaxBayesianQLearner
from agents.qlearner import JaxQLearner
import axelrod as axl
import numpy as np
import os
import itertools
import random
import tqdm

# Environment parameters
noise_levels = list(np.arange(0, 0.30, 0.05).round(2))
repetitions = 10
turns = 1000
processes = 32
seed = 42

strategies = [
    axl.DBS(0.999, 5, 2, 3, 5),
    axl.DBS(0.75, 5, 4, 5, 5),
    JaxFiveStateAgent(
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
    JaxFiveStateAgentNoisy(
        pB_scale=1,
        gamma=1,
        alpha=1,
        bias=0.5,
        preference="standard",
        policy_len=10,
        update_interval=10,
        seed=seed,
        lr_B=1,
    ),
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
    JaxQLearner(
        learning_rate=0.9,
        discount_rate=0.9,
        action_selection_parameter=0.1,
    ),
    JaxBayesianQLearner(
        discount_rate=0.9,
        initial_variance=1.0,
        reward_variance=1.0,
    ),
]

def run_experiment(strategies, opponents, dir_name):
    for i, strategy in enumerate(strategies):
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

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_experiment(strategies, static_pool, "static")
    run_experiment(strategies, learn_pool, "learning")