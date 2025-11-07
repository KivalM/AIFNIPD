from tournament.handler import FileSystemHandler
from tournament.tournament import NoiseTournament
from experiments import static_pool, learn_pool
import itertools

import axelrod as axl
import numpy as np
from agents.psrl import PSRL
import os
import random
import tqdm

# Environment parameters
noise_levels = list(np.arange(0, 0.30, 0.05).round(2))
repetitions = 5
turns = 1000
processes = 16
seed = 42
np.random.seed(seed)
random.seed(seed)

# Agent parameters (12 Combinations)
prior_strengths = [0.5, 1.0, 2.0]
discount_rates = [0.5, 0.9]
value_iteration_steps = [50, 100]

def run_search():
    # Create all combinations
    combinations = list(itertools.product(prior_strengths, discount_rates, value_iteration_steps))
    print(f"Running {len(combinations)} combinations for PSRL")
    
    for combination in combinations:
        prior_strength, discount_rate, value_iteration_step = combination
        agent = PSRL(
            prior_strength=prior_strength,
            discount_rate=discount_rate,
            value_iteration_steps=value_iteration_step
        )
        agent.name = f"PSRL_{prior_strength}_{discount_rate}_{value_iteration_step}"
        strats = [agent] + static_pool + learn_pool
        handler = FileSystemHandler(root_dir=f"results/hyperparameters/psrl/{agent.name}")
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

