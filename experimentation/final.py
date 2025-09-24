# %%
import os 
import multiprocessing
multiprocessing.freeze_support()

# %%
import axelrod as axl
import numpy as np
from agents.aif.jax.factorized import JaxFactorizedAgent
from agents.aif.jax.five_state_noise import JaxFiveStateAgentNoisy
from agents.aif.jax.five_state import JaxFiveStateAgent

# Generate noise levels from 0 to 0.5 at 0.05 intervals
noise_levels = list(np.arange(0, 0.50, 0.05).round(2))

testing_strategies = [
    axl.DBS(),
    axl.RiskyQLearner(),
    axl.ArrogantQLearner(),
    axl.CautiousQLearner(),
    axl.HesitantQLearner(),
    JaxFiveStateAgent(
        10, 10, 5, 1.5, 0.6, 1.0, 0.5, "nash", 1
    ),
    JaxFiveStateAgent(
        10, 50, 5, 1, 1, 1.0, 0.25, "nash", 1
    ),
    JaxFactorizedAgent(
        10, 50, 5, 0.5, 0.3, 1.0, False, 0.75, "nash", 10
    ),
    JaxFactorizedAgent(
        10, 100, 5, 1.5, 0.6, 0.6, False, 0.50, "nash", 100
    ),
    axl.APavlov2011(),
    axl.AdaptiveTitForTat(),
    axl.GTFT(),
    axl.ContriteTitForTat(),
    axl.StochasticWSLS(),
    axl.WinStayLoseShift(),
    axl.Cooperator(),
    axl.Defector(),
    JaxFiveStateAgent(10, 50, 5, 1, 1, 1.0, 0.5, "nash", 1),

]

strategies = [
    axl.Cooperator(),
    axl.Defector(),
    axl.TitForTat(),
    axl.Random(),
    axl.GTFT(),
    axl.Grudger(),
    axl.ZDExtort4(),
    axl.ZDGen2(),
    axl.RiskyQLearner(),
    axl.ArrogantQLearner(),
    axl.CautiousQLearner(),
    axl.EvolvedANNNoise05(),
    axl.OmegaTFT(),
    axl.DBS(),
    axl.ContriteTitForTat(),
    axl.Random(),
    axl.WinStayLoseShift()
]



# %%
def main():
    from tournament.handler import FileSystemHandler
    from tournament.tournament import NoiseTournament
    import os

    for i, strategy in enumerate(testing_strategies):
        repetitions = 10
        seed = 42
        turns = 1500
        processes = os.cpu_count() - 4

        players = [strategy] + strategies

        # # 2. Instantiate the handler
        handler = FileSystemHandler(root_dir=f"final/{i}_{strategy.__class__.__name__}")

        # # 3. Instantiate the tournament with the handler's callbacks
        noise_tournament = NoiseTournament(
            players=players,
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

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()


