import json
import os
from pathlib import Path
from typing import Any, Dict, List

import axelrod as axl
import numpy as np

from .tournament import NoiseTournament


class FileSystemHandler:
    """
    Handles saving tournament results to the filesystem and skipping existing runs.
    """

    def __init__(self, root_dir: str = "results"):
        self.root_dir = Path(root_dir)

    def _get_paths(self, config: Dict[str, Any]) -> (Path, Path, Path, Path):
        """Helper to generate paths for a given configuration."""
        repetition_dir = self.root_dir / f"repetition_{config['repetition']}"
        details_file = repetition_dir / "details.json"
        json_results_file = repetition_dir / f"{config['noise']}.json"
        csv_results_file = repetition_dir / f"{config['noise']}.csv"
        return repetition_dir, details_file, json_results_file, csv_results_file

    def skip_run(self, config: Dict[str, Any]) -> bool:
        """
        Checks if the results for a given configuration already exist.
        To be used as the `skip_callback` for NoiseTournament.
        """
        _, _, _, csv_results_file = self._get_paths(config)
        if csv_results_file.exists():
            print(f"Skipping existing run: {csv_results_file}")
            return True
        return False

    def save_results(
        self,
        results: axl.ResultSet,
        config: Dict[str, Any],
        intended_histories: Dict[int, Dict[int, List[str]]],
    ):
        """
        Saves the results of a tournament to JSON files.
        To be used as the `callback` for NoiseTournament.
        """
        (
            repetition_dir,
            details_file,
            json_results_file,
            csv_results_file,
        ) = self._get_paths(config)
        repetition_dir.mkdir(parents=True, exist_ok=True)

        if not details_file.exists():
            details = {
                "seed": int(config["seed"]),
                "turns": int(config["turns"]),
                "player_names_hash": config["player_names_hash"],
                "player_names": results.players,
            }
            with open(details_file, "w") as f:
                json.dump(details, f, indent=4)

        df = results.df

        # --- Add Intended Actions and calculate noise events ---
        player_indices = {name: i for i, name in enumerate(results.players)}
        def get_intended_actions(row):
            p_index = player_indices[row["Player name"]]
            o_index = player_indices[row["Opponent name"]]
            if len(intended_histories[p_index][o_index]) > 0:
                return intended_histories[p_index][o_index].pop(0)
            else:
                return None

        df["Intended Actions"] = df.apply(get_intended_actions, axis=1)


        df.to_csv(csv_results_file, index=False)

        stats: Dict[str, Any] = {}
        for i, player_name in enumerate(results.players):
            player_df = df[df["Player name"] == player_name].copy()
            player_noise_events = (
                player_df["Actions"] != player_df["Intended Actions"]
            ).sum()

            std_score_raw = player_df["Score"].std()
            std_score = (
                float(std_score_raw) if not np.isnan(std_score_raw) else None
            )

            stats[player_name] = {
                "mean_score": float(player_df["Score"].mean()),
                "std_score": std_score,
                "total_score": float(player_df["Score"].sum()),
                "wins": int(player_df["Win"].sum()),
                "cooperation_rate": float(results.cooperating_rating[i]),
                "noise_events": int(player_noise_events),
            }

        with open(json_results_file, "w") as f:
            json.dump(stats, f, indent=4)
        # print(f"Saved results to {csv_results_file} and {json_results_file}")


def main():
    """
    Example of how to run the NoiseTournament with the FileSystemHandler.
    """
    print("Setting up and running a test tournament...")

    # 1. Define players and tournament parameters
    players = [
        axl.Random(),
        axl.TitFor2Tats(),
    ]


    noise_levels = [0]
    repetitions = 1
    seed = 42
    processes = os.cpu_count()

    # 2. Instantiate the handler
    handler = FileSystemHandler(root_dir="tournament_results")

    # 3. Instantiate the tournament with the handler's callbacks
    noise_tournament = NoiseTournament(
        players=players,
        noise_levels=noise_levels,
        repetitions=repetitions,
        seed=seed,
        callback=handler.save_results,
        # skip_callback=handler.skip_run,
    )

    # 4. Run the tournament
    noise_tournament.run(turns=10, processes=processes)

    print("Test tournament finished.")
    print(f"Results are saved in the '{handler.root_dir}' directory.")


if __name__ == "__main__":
    main()
