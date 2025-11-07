from tournament.handler import FileSystemHandler
from tournament.tournament import NoiseTournament
from experiments import static_pool, learn_pool
import itertools

import axelrod as axl
import numpy as np
import os
import random
import tqdm

# Import DynaQ from hyphenated filename
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("dynaq_module", "src/agents/dyna-Q.py")
dynaq_module = importlib.util.module_from_spec(spec)
sys.modules["dynaq_module"] = dynaq_module
spec.loader.exec_module(dynaq_module)
DynaQ = dynaq_module.DynaQ

# Environment parameters
noise_levels = list(np.arange(0, 0.30, 0.05).round(2))
repetitions = 5
turns = 1000
processes = 16
seed = 42
np.random.seed(seed)
random.seed(seed)

# Agent parameters (18 Combinations)
learning_rates = [0.1, 0.5, 0.9]
discount_rates = [0.5, 0.9]
action_selection_parameters = [0.1]
planning_steps = [5, 20]

def run_search():
    # Create all combinations
    combinations = list(itertools.product(learning_rates, discount_rates, action_selection_parameters, planning_steps))
    print(f"Running {len(combinations)} combinations for Dyna-Q")
    
    for combination in combinations:
        learning_rate, discount_rate, action_selection_parameter, planning_step = combination
        agent = DynaQ(learning_rate, discount_rate, action_selection_parameter, planning_step)
        agent.name = f"DynaQ_{learning_rate}_{discount_rate}_{action_selection_parameter}_{planning_step}"
        strats = [agent] + static_pool + learn_pool
        handler = FileSystemHandler(root_dir=f"results/hyperparameters/dynaq/{agent.name}")
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

