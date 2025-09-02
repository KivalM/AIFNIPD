from __future__ import annotations

from typing import Dict, Optional, Tuple

from axelrod import Player, Action


C, D = Action.C, Action.D


class JointWrapper(Player):
    """Base wrapper to expose a joint-state `.step` API for Axelrod players.

    Subclasses should implement `step(state)` where `state` is the joint
    state for the current decision: `(my_last_action, opponent_last_action)`.
    The initial call uses `(None, None)`.
    """

    name = "JointWrapper"
    classifier = {
        "memory_depth": 2,
        "stochastic": True,
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self) -> None:
        super().__init__()
        self.classifier["stochastic"] = True

        # Joint state tracking
        self.prev_state: Tuple[Optional[Action], Optional[Action]] = (None, None)
        self.prev_action: Optional[Action] = None

        # Payoff matrix is set when a match starts
        self.payoff_matrix: Dict[Action, Dict[Action, float]] = {
            C: {C: 0.0, D: 0.0},
            D: {C: 0.0, D: 0.0},
        }

    def receive_match_attributes(self) -> None:  # type: ignore[override]
        (R, P, S, T) = self.match_attributes["game"].RPST()
        self.payoff_matrix = {C: {C: R, D: S}, D: {C: T, D: P}}

    def strategy(self, opponent: Player) -> Action:  # type: ignore[override]
        if not self.history:
            state = (None, None)
        else:
            state = (self.prev_action, opponent.history[-1])

        action = self.step(state)

        # Advance wrapper state for next round
        self.prev_state = state
        self.prev_action = action
        return action

    def step(self, state: Tuple[Optional[Action], Optional[Action]]) -> Action:
        """Return the action to take given the joint `state`.

        Subclasses must implement this method.
        """
        raise NotImplementedError
