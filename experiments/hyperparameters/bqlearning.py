from tournament.handler import FileSystemHandler
from tournament.tournament import NoiseTournament
from experiments import static_pool, learn_pool
import itertools

import axelrod as axl
import numpy as np
from agents.bqlearner import JaxBayesianQLearner
import os
import itertools
import random
import tqdm

# Environment parameters
noise_levels = list(np.arange(0, 0.30, 0.05).round(2))
repetitions = 5
turns = 1000
processes = os.cpu_count() - 2
seed = 42
np.random.seed(seed)
random.seed(seed)

# Agent parameters (18 Combinations)
discount_rates = [0.1, 0.5, 0.9]
initial_variances = [0.1, 0.5, 1.0]
reward_variances = [0.1, 0.5, 1.0]

def run_search():
    # Create all combinations
    combinations = list(itertools.product(discount_rates, initial_variances, reward_variances))

    for combination in combinations:
        discount_rate, initial_variance, reward_variance = combination
        agent = JaxBayesianQLearner(discount_rate, initial_variance, reward_variance)
        agent.name = f"BQLearner_{discount_rate}_{initial_variance}_{reward_variance}"
        strats = [agent] + static_pool + learn_pool
        handler = FileSystemHandler(root_dir=f"results/hyperparameters/bqlearning/{agent.name}")
        noise_tournament = NoiseTournament(
            players=strats,
            noise_levels=noise_levels,
            repetitions=repetitions,
            seed=seed,
            callback=handler.save_results,
            skip_callback=handler.skip_run,
        )
        noise_tournament.run(turns=turns, processes=processes)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    run_search()