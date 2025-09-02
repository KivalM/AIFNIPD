import axelrod as axl
from typing import Callable

def run_experiment(strategy: Callable[[], axl.Player], turns: int = 1000, noise_level: float = 0.0):
    others = [
        axl.Cooperator,
        axl.Defector,
        axl.TitForTat,
        axl.WinStayLoseShift,
        axl.DBS,
        axl.ZDExtortion,
    ]

    results = []
    for other in others:
        match = axl.Tournament((strategy(), other()), turns=turns, noise=noise_level)
        results.append(match.play())
    return results

def run_experiment_for_all_strategies(turns: int = 1000, noise_level: float = 0.0):
    strategies = [
        axl.DBS,
        axl.ZDExtortion,
    ]
    for strategy in strategies:
        results = run_experiment(strategy, turns, noise_level)
        print(results.summary())

if __name__ == "__main__":
    run_experiment_for_all_strategies()