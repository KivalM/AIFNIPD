from tournament.handler import FileSystemHandler
from tournament.tournament import NoiseTournament
import itertools

import axelrod as axl
import numpy as np
from agents.aif.jax.aif import ActiveInferenceAgent
from agents.aif.jax.epsilon_greedy import EpsilonGreedyAIFAgent
import os
import itertools
import random
import tqdm

# Environment parameters
noise_levels = list(np.arange(0, 0.50, 0.05).round(2))
repetitions = 1
turns = 1000
processes = 25
seed = 42
np.random.seed(seed)
random.seed(seed)

# Agent parameters (18 Combinations)
pB_lrs = [1]
c_preferences = [False]
noisy_A_options = [False, True]
policy_lens = [1, 3, 5, 10]
update_intervals = [5, 10, 20, 50]
pB_scales = [1]
gammas = [1]
alphas = [1]
biases = [0.5]
# Epsilon-greedy parameters
epsilon_starts = [0.5, 1.0]
epsilon_decays = [0.99, 0.995, 0.999]
epsilon_mins = [0.01]

        # pB_scale=1,
        # gamma=1,
        # alpha=1,
        # bias=0.5,
        # cooperative_preference=False,
        # policy_len=5,
        # update_interval=10,
        # seed=seed,
        # lr_B=1,
        # action_selection="deterministic",
        # use_noisy_observation_model=True,
        # use_states_info_gain=False,
        # use_param_info_gain=False,

def run_search():
    # Create all combinations
    combinations = list(itertools.product(pB_lrs, c_preferences, noisy_A_options, policy_lens, update_intervals, pB_scales, gammas, alphas, biases))

    for combination in tqdm.tqdm(combinations):
        pB_lr, c_preference, noisy_A_option, policy_len, update_interval, pB_scale, gamma, alpha, bias = combination
        agent = ActiveInferenceAgent(
            lr_B=pB_lr,
            cooperative_preference=c_preference,
            policy_len=policy_len,
            update_interval=update_interval,
            pB_scale=pB_scale,
            gamma=gamma,
            alpha=alpha,
            bias=bias,
            use_noisy_observation_model=noisy_A_option,
        )
        agent.name = f"AIF_{pB_lr}_{c_preference}_{noisy_A_option}_{policy_len}_{update_interval}_{pB_scale}_{gamma}_{alpha}_{bias}"
        strats = [agent, axl.TitForTat()] 
        handler = FileSystemHandler(root_dir=f"results/paper/hyperparameters/aif/{agent.name}")
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