import hashlib
import multiprocessing
import os
import tempfile
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import axelrod as axl
import dask.dataframe as dd
import numpy as np
from axelrod.match_generator import MatchChunk
from tqdm import tqdm


class CustomMatch(axl.Match):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intended_actions_pair = []

    def simultaneous_play(self, player, coplayer, noise=0):
        s1, s2 = player.strategy(coplayer), coplayer.strategy(player)
        self.intended_actions_pair.append((s1, s2))
        if noise:
            s1 = self._random.random_flip(s1, noise)
            s2 = self._random.random_flip(s2, noise)
        player.update_history(s1, s2)
        coplayer.update_history(s2, s1)
        return s1, s2


class CustomTournament(axl.Tournament):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intended_histories = defaultdict(lambda: defaultdict(list))

    def _play_matches(self, chunk: MatchChunk, build_results: bool = True):
        interactions = defaultdict(list)
        p1_index, p2_index = chunk.index_pair
        player1 = self.players[p1_index].clone()
        player2 = self.players[p2_index].clone()

        chunk.match_params["players"] = (player1, player2)
        chunk.match_params["seed"] = chunk.seed
        match = CustomMatch(**chunk.match_params)
        for _ in range(chunk.repetitions):
            match.play()

            if build_results:
                results = self._calculate_results(match.result)
            else:
                results = None

            interactions[chunk.index_pair].append([match.result, results])
            intended_actions1 = [i[0] for i in match.intended_actions_pair]
            intended_actions2 = [i[1] for i in match.intended_actions_pair]

            self.intended_histories[p1_index][p2_index].append(
                "".join(str(a) for a in intended_actions1)
            )
            self.intended_histories[p2_index][p1_index].append(
                "".join(str(a) for a in intended_actions2)
            )
        return interactions


# Worker function for multiprocessing
def _run_single_tournament_worker(
    config: Dict[str, Any]
) -> (Optional[axl.ResultSet], Dict[str, Any], Dict):
    """
    A top-level function to run a single tournament.
    Required for multiprocessing.
    """
    players = [p.clone() for p in config["players"]]

    tour = CustomTournament(
        players=players,
        noise=config["noise"],
        turns=config["turns"],
        repetitions=1,  # We handle repetitions externally
        seed=config["seed"],
    )

    # We need to manage a temporary file to get the raw results,
    # as ResultSet does not store the dataframe.
    fd, filename = tempfile.mkstemp(suffix=".csv")
    os.close(fd)

    try:
        # Run silently in worker processes, writing to the temp file
        results = tour.play(
            build_results=True, progress_bar=False, filename=filename
        )
        # Read the raw results and attach them to the ResultSet object
        results.df = dd.read_csv(filename).compute()
    finally:
        # Ensure the temporary file is removed
        os.remove(filename)

    # Convert defaultdicts to plain dicts for serialization
    intended_histories_dict = {
        p_idx: dict(opp_hist) for p_idx, opp_hist in tour.intended_histories.items()
    }
    return results, config, intended_histories_dict


class NoiseTournament:
    """
    A class to run a series of Axelrod tournaments with varying noise levels
    and repetitions.
    """

    def __init__(
        self,
        players: List[axl.Player],
        noise_levels: List[float],
        repetitions: int,
        seed: int,
        callback: Optional[
            Callable[[axl.ResultSet, Dict[str, Any], Dict], None]
        ] = None,
        skip_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ):
        self.players = players
        self.noise_levels = noise_levels
        self.repetitions = repetitions
        self.callback = callback
        self.skip_callback = skip_callback
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        player_names = sorted([str(p) for p in self.players])
        player_string = "".join(player_names)
        self.player_names_hash = hashlib.md5(
            player_string.encode("utf-8")
        ).hexdigest()

    def run(self, turns: int = 200, processes: Optional[int] = None):
        """
        Runs the tournaments.

        Parameters
        ----------
        turns : int
            The number of turns per match.
        processes : int, optional
            The number of processes to use for parallel execution.
            If None or 1, runs serially. Defaults to None.
        """

        repetition_seeds = self.rng.integers(0, 2**32 - 1, self.repetitions)

        configs_to_run = []
        for rep_idx in range(self.repetitions):
            for noise in self.noise_levels:
                config = {
                    "repetition": rep_idx,
                    "noise": noise,
                    "seed": repetition_seeds[rep_idx],
                    "turns": turns,
                    "players": self.players,
                    "player_names_hash": self.player_names_hash,
                }

                if self.skip_callback and self.skip_callback(config):
                    continue

                configs_to_run.append(config)

        if processes and processes > 1:
            with multiprocessing.Pool(processes) as pool:
                pbar = tqdm(total=len(configs_to_run))
                for result, config, intended_histories in pool.imap_unordered(
                    _run_single_tournament_worker, configs_to_run
                ):
                    if self.callback:
                        self.callback(result, config, intended_histories)
                    pbar.update()
                pbar.close()
        else:
            # Serial execution
            for config in tqdm(configs_to_run):
                (
                    result,
                    returned_config,
                    intended_histories,
                ) = _run_single_tournament_worker(config)
                if self.callback:
                    self.callback(result, returned_config, intended_histories)