from tournament.handler import FileSystemHandler
from tournament.tournament import NoiseTournament
from experiments import static_pool, learn_pool
import itertools

import axelrod as axl
import numpy as np
from agents.aif.jax.five_state import JaxFiveStateAgent
from agents.aif.jax.five_state_noise import JaxFiveStateAgentNoisy
from agents.aif.jax.five_state_deterministic_noise import JaxFiveStateAgentDeterministicNoisy
from agents.aif.jax.five_state_deterministic import JaxFiveStateAgentDeterministic
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
pB_lrs = [0.5, 1, 1.5]
c_preferences = ["standard", "nash"]
noisy_A_options = [True, False]
policy_lens = [1, 10]
update_intervals = [10, 50, 100]
pB_scales = [1]
gammas = [1]
alphas = [1]
biases = [0.5]

def run_search():
    # Create all combinations
    combinations = list(itertools.product(pB_lrs, c_preferences, noisy_A_options, policy_lens, update_intervals, pB_scales, gammas, alphas, biases))

    for combination in combinations:
        pB_lr, c_preference, noisy_A_option, policy_len, update_interval, pB_scale, gamma, alpha, bias = combination
        if noisy_A_option:
            agent = JaxFiveStateAgentDeterministicNoisy(
                lr_B=pB_lr,
                preference=c_preference,
                policy_len=policy_len,
                update_interval=update_interval,
                pB_scale=pB_scale,
                gamma=gamma,
                alpha=alpha,
                bias=bias
            )   
        else:   
            agent = JaxFiveStateAgentDeterministic(
                lr_B=pB_lr,
                preference=c_preference,
                policy_len=policy_len,
                update_interval=update_interval,
                pB_scale=pB_scale,
                gamma=gamma,
                alpha=alpha,
                bias=bias
            )
        agent.name = f"AIF_{pB_lr}_{c_preference}_{noisy_A_option}_{policy_len}_{update_interval}_{pB_scale}"
        strats = [agent] + static_pool + learn_pool
        handler = FileSystemHandler(root_dir=f"results/hyperparameters/actinf/{agent.name}")
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