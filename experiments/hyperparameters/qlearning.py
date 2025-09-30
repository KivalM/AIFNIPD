from tournament.handler import FileSystemHandler
from tournament.tournament import NoiseTournament
from experiments import static_pool, learn_pool
import itertools

import axelrod as axl
import numpy as np
from agents.qlearner import JaxQLearner
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
learning_rates = [0.1, 0.5, 0.9]
discount_rates = [0.1, 0.5, 0.9]
action_selection_parameters = [0, 0.1]

def run_search():
    # Create all combinations
    combinations = list(itertools.product(learning_rates, discount_rates, action_selection_parameters))

    for combination in combinations:
        learning_rate, discount_rate, action_selection_parameter = combination
        agent = JaxQLearner(learning_rate, discount_rate, action_selection_parameter)
        agent.name = f"QLearner_{learning_rate}_{discount_rate}_{action_selection_parameter}"
        strats = [agent] + static_pool + learn_pool
        handler = FileSystemHandler(root_dir=f"results/hyperparameters/qlearning/{agent.name}")
        noise_tournament = NoiseTournament(
            players=strats,
            noise_levels=noise_levels,
            repetitions=repetitions,
            seed=seed,
            callback=handler.save_results,
            skip_callback=handler.skip_run,
        )
        noise_tournament.run(turns=turns, processes=processes)