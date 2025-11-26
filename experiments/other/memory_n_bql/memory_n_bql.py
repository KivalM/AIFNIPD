from typing import Any


from tournament.handler import FileSystemHandler
from tournament.tournament import NoiseTournament
import multiprocessing
from agents.bqlearner import JaxBayesianQLearner
import axelrod as axl
import numpy as np
from experiments import static_pool, learn_pool

# Environment parameters
noise_levels: list[Any] = list[Any](np.arange(0, 0.15, 0.05).round(2))
repetitions: int = 10
turns: int = 1000
processes: int = 1
seed: int = 42

strategies = [
    JaxBayesianQLearner(
        discount_rate=0.5,
        initial_variance=1.0,
        reward_variance=1.0,
        memory_length=m,
    )
    for m in range(1, 9)
]

# Define games
games = {
    "pd": axl.Game(r=3, s=0, t=5, p=1),
    # "chicken": axl.Game(r=3, s=1, t=4, p=0),
    # "stag": axl.Game(r=4, s=0, t=3, p=3),
}

def run_self_play_experiment(strategies: list[JaxBayesianQLearner]):
    """Run a single round-robin tournament with all strategies."""
    for game_name, game in games.items():
        print(f"\nRunning self-play experiments for {game_name}...")
        handler = FileSystemHandler(root_dir=f"results/other/memory_n_bql/{game_name}")
        noise_tournament = NoiseTournament(
            players=strategies,  # All strategies play against each other in round-robin
            noise_levels=noise_levels,
            repetitions=repetitions,
            seed=seed,
            game=game,
            callback=handler.save_results,
            skip_callback=handler.skip_run,
        )
        noise_tournament.run(turns=turns, processes=processes)
        print(f"Results for {game_name} are saved in the '{handler.root_dir}' directory.")



if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    run_self_play_experiment(strategies)

