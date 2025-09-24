# %%
from tournament.handler import FileSystemHandler
from tournament.tournament import NoiseTournament
import itertools

# %%
import axelrod as axl
import numpy as np
from agents.aif.jax.five_state import JaxFiveStateAgent
from agents.aif.jax.five_state_noise import JaxFiveStateAgentNoisy
from agents.aif.jax.factorized import JaxFactorizedAgent
import os
import itertools
import random
import tqdm

# %%
noise_levels = list(np.arange(0, 0.30, 0.05).round(2))
pB_scales = [1, 10, 100]
pB_lrs = [0.5, 1, 1.5]
coop_biases = [0.25, 0.5, 0.75]
gammas = [0.3, 0.6, 1]
alphas = [0.3, 0.6, 1]
policy_lens = [1, 5, 10]
update_intervals = [10, 50, 100]
c_preferences = ["standard", "nash"]
noisy_A_options = [True, False]

# %%
repetitions = 4
turns = 1000
processes = os.cpu_count() - 2
seed = 42
num_samples = 30
np.random.seed(seed)
random.seed(seed)

# %%
default_pool = [
    axl.DBS(),
    axl.TitForTat(),
    axl.Random(),
    axl.Cooperator(),
    axl.Defector(),
    axl.Grudger(),
    axl.ZDExtort2(),
    axl.ContriteTitForTat(),
    axl.OmegaTFT(),
    axl.Random(),
]

# %%
def run_factorized():
    factorized_params = list(itertools.product(pB_scales, pB_lrs, coop_biases, gammas, alphas, policy_lens, update_intervals, c_preferences, noisy_A_options))
    chosen = random.sample(factorized_params, k=num_samples)

    # %%
    # Param Sweep for Factorized
    for pB_scale, \
        pB_lr, \
        coop_bias, \
        gamma, \
        alpha, \
        policy_len, \
        update_interval, \
        preferences, \
        noisy_A \
    in tqdm.tqdm(chosen):
        try:
            player = JaxFactorizedAgent(
                seed=0, 
                policy_len=policy_len,
                update_interval=update_interval,
                lr_B=pB_lr, 
                alpha=alpha, 
                gamma=gamma,
                noisy_A=noisy_A,
                bias=coop_bias,
                preference=preferences,
                pB_scale=pB_scale,
            )
            player.name = f"AIF_FACTORIZED_{pB_scale}_{pB_lr}_{coop_bias}_{gamma}_{alpha}_{policy_len}_{update_interval}_{preferences}_{noisy_A}"
            strats = [player] + default_pool
            # # 2. Instantiate the handler
            handler = FileSystemHandler(root_dir=f"jax_hyperparameter_tuning3/{player.name}")

            # # 3. Instantiate the tournament with the handler's callbacks
            noise_tournament = NoiseTournament(
                players=strats,
                noise_levels=noise_levels,
                repetitions=repetitions,
                seed=seed,
                callback=handler.save_results,
                skip_callback=handler.skip_run,
            )

            # # 4. Run the tournament
            noise_tournament.run(turns=turns, processes=processes)

            # print("Test tournament finished.")
            print(f"Results are saved in the '{handler.root_dir}' directory.")
        except Exception as e:
            raise e
            continue


def run_joint():
# %%
    joint_params = list(itertools.product(pB_scales, pB_lrs, coop_biases, gammas, alphas, policy_lens, update_intervals, c_preferences, noisy_A_options))
    chosen = random.sample(joint_params, k=num_samples)

    # %%
    # Param Sweep for Joint
    for pB_scale, \
        pB_lr, \
        coop_bias, \
        gamma, \
        alpha, \
        policy_len, \
        update_interval, \
        preferences, \
        noisy_A \
    in tqdm.tqdm(chosen):
        try:
            player = None
            if noisy_A:
                player = JaxFiveStateAgentNoisy(
                    seed=0, 
                    policy_len=policy_len,
                    update_interval=update_interval,
                    lr_B=pB_lr, 
                    alpha=alpha, 
                    gamma=gamma,
                    bias=coop_bias,
                    preference=preferences,
                    pB_scale=pB_scale,
                )
            else:
                player = JaxFiveStateAgent(
                    seed=0, 
                    policy_len=policy_len,
                    update_interval=update_interval,
                    lr_B=pB_lr, 
                    alpha=alpha, 
                    gamma=gamma,
                    bias=coop_bias,
                    preference=preferences,
                    pB_scale=pB_scale,
                )
            
            player.name = f"AIF_Joint_{pB_scale}_{pB_lr}_{coop_bias}_{gamma}_{alpha}_{policy_len}_{update_interval}_{preferences}_{noisy_A}"
            strats = [player] + default_pool
            # # 2. Instantiate the handler
            handler = FileSystemHandler(root_dir=f"jax_hyperparameter_tuning3/{player.name}")

            # # 3. Instantiate the tournament with the handler's callbacks
            noise_tournament = NoiseTournament(
                players=strats,
                noise_levels=noise_levels,
                repetitions=repetitions,
                seed=seed,
                callback=handler.save_results,
                skip_callback=handler.skip_run,
            )

            # # 4. Run the tournament
            noise_tournament.run(turns=turns, processes=processes)

            # print("Test tournament finished.")
            print(f"Results are saved in the '{handler.root_dir}' directory.")
        except Exception as e:
            print(f"Error with {player.name}: {e}")
            continue



if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    run_factorized()
    run_joint()