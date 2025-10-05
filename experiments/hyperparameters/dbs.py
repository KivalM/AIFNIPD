from tournament.handler import FileSystemHandler
from tournament.tournament import NoiseTournament
from experiments import static_pool, learn_pool
import itertools

import axelrod as axl
import numpy as np
import os
import itertools
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

# Agent parameters (18 Combinations)
        # discount_factor=0.75,
        # promotion_threshold=3,
        # violation_threshold=4,
        # reject_threshold=3,
        # tree_depth=5,
discount_factors = [0.75, 0.999]
promotion_thresholds = [3, 5]
violation_thresholds = [2, 4]
reject_thresholds = [3, 5]
tree_depths = [3,5]

def run_search():
    # Create all combinations
    combinations = list(itertools.product(discount_factors, promotion_thresholds, violation_thresholds, reject_thresholds, tree_depths))
    print(f"Running {len(combinations)} combinations")
    for combination in combinations:
        discount_factor, promotion_threshold, violation_threshold, reject_threshold, tree_depth = combination
        agent = axl.DBS(discount_factor, promotion_threshold, violation_threshold, reject_threshold, tree_depth)
        agent.name = f"DBS_{discount_factor}_{promotion_threshold}_{violation_threshold}_{reject_threshold}_{tree_depth}"
        strats = [agent] + static_pool + learn_pool
        handler = FileSystemHandler(root_dir=f"results/hyperparameters/dbs/{agent.name}")
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