from tournament.handler import FileSystemHandler
from tournament.tournament import NoiseTournament
import multiprocessing
from agents.aif.jax.five_state import JaxFiveStateAgent
from agents.aif.jax.five_state_noise import JaxFiveStateAgentNoisy
from agents.aif.jax.five_state_utility import JaxFiveStateAgentUtility
from agents.bqlearner import JaxBayesianQLearner, CooperativeBQLearner
from agents.qlearner import JaxQLearner
from agents.qlearner import CooperativeQLearner
import axelrod as axl
import numpy as np

# Environment parameters
noise_levels = list(np.arange(0, 0.30, 0.05).round(2))
repetitions = 30
turns = 1000
processes = 20
seed = 42

strategies = [
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
    JaxQLearner(
        learning_rate=0.9,
        discount_rate=0.9,
        action_selection_parameter=0.1,
    ),
    CooperativeQLearner(
        learning_rate=0.9,
        discount_rate=0.9,
        action_selection_parameter=0.1,
    ),

    JaxBayesianQLearner(
        discount_rate=0.5,
        initial_variance=1.0,
        reward_variance=1.0,
    ),
    CooperativeBQLearner(
        discount_rate=0.5,
        initial_variance=1.0,
        reward_variance=1.0,
    ),
    JaxFiveStateAgentNoisy(
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
    JaxFiveStateAgentNoisy(
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
    #
    axl.DBS(0.999, 5, 2, 3, 5),
    axl.GTFT(),
    axl.ContriteTitForTat(),
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
    JaxFiveStateAgentUtility(
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
]

def run_self_play_experiment(strategies):
    """Run a single round-robin tournament with all strategies."""
    handler = FileSystemHandler(root_dir="results/sp")
    noise_tournament = NoiseTournament(
        players=strategies,  # All strategies play against each other in round-robin
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
    
    # Run self-play experiments
    run_self_play_experiment(strategies)

