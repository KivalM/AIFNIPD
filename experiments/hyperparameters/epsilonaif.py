from tournament.handler import FileSystemHandler
from tournament.tournament import NoiseTournament
from experiments import static_pool, learn_pool
import itertools

import numpy as np
from agents.aif.jax.epsilon_greedy import EpsilonGreedyAIFAgent
import random

# Environment parameters
noise_levels = list(np.arange(0, 0.10, 0.05).round(2))
repetitions = 5
turns = 1000
processes = 16
seed = 42
np.random.seed(seed)
random.seed(seed)

# AIF base parameters
pB_lrs = [1]
cooperative_preferences = [False]
policy_lens = [5]
update_intervals = [10]
pB_scales = [1]
biases = [0.5]

# Epsilon-greedy parameters
epsilon_starts = [0.5, 1.0]
epsilon_decays = [0.99, 0.995, 0.999]
epsilon_mins = [0.01]


def run_search():
    combinations = list(itertools.product(
        pB_lrs, cooperative_preferences, policy_lens, update_intervals,
        pB_scales, biases, epsilon_starts, epsilon_decays, epsilon_mins,
    ))

    for combo in combinations:
        (pB_lr, coop_pref, policy_len, update_interval,
         pB_scale, bias, eps_start, eps_decay, eps_min) = combo

        agent = EpsilonGreedyAIFAgent(
            lr_B=pB_lr,
            cooperative_preference=coop_pref,
            policy_len=policy_len,
            update_interval=update_interval,
            pB_scale=pB_scale,
            bias=bias,
            epsilon_start=eps_start,
            epsilon_min=eps_min,
            epsilon_decay=eps_decay,
        )
        coop_label = "coop" if coop_pref else "std"
        agent.name = (
            f"EpsAIF_{pB_lr}_{coop_label}_{policy_len}"
            f"_{update_interval}_{pB_scale}"
            f"_e{eps_start}_d{eps_decay}_m{eps_min}"
        )
        strats = [agent] + static_pool + learn_pool
        handler = FileSystemHandler(root_dir=f"results/hyperparameters/epsilonaif/{agent.name}")
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
