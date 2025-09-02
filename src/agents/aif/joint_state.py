"""
Active Inference agent using a joint-state representation with the shared
JointWrapper `.step(state)` interface. The joint state and observation spaces
are S = O = {CC, CD, DC, DD}.
"""

from __future__ import annotations

from typing import Optional, Tuple

import axelrod
import numpy as np
from axelrod.action import Action
from pymdp.agent import Agent
import pymdp.utils as utils
from ..wrapper import C, JointWrapper

# Indices for AIF model spaces
STATE_CC_IDX = 0
STATE_CD_IDX = 1
STATE_DC_IDX = 2
STATE_DD_IDX = 3
ALL_STATE_IDXS = [STATE_CC_IDX, STATE_CD_IDX, STATE_DC_IDX, STATE_DD_IDX]

ACTION_C_IDX = 0
ACTION_D_IDX = 1


def ego_A(noise: float = 0.2) -> np.ndarray:
    A = np.zeros((len(ALL_STATE_IDXS), len(ALL_STATE_IDXS)))
    for s in ALL_STATE_IDXS:
        A[:, s] = noise / 3.0
        A[s, s] = 1.0 - noise
    return A


def ego_B() -> np.ndarray:
    # B[s_curr, s_next, a]
    B = np.ones((len(ALL_STATE_IDXS), len(ALL_STATE_IDXS), 2))
    B = B / B.sum(axis=1, keepdims=True)
    return B


def ego_C() -> np.ndarray:
    # Preferences over observations (R, P, S, T ordering encoded)
    C_vec = np.zeros(len(ALL_STATE_IDXS))
    C_vec[STATE_CC_IDX] = 3
    C_vec[STATE_CD_IDX] = 0
    C_vec[STATE_DC_IDX] = 4
    C_vec[STATE_DD_IDX] = 1
    return C_vec


def ego_D() -> np.ndarray:
    D_vec = np.ones(len(ALL_STATE_IDXS))
    D_vec = D_vec / D_vec.sum()
    return D_vec


def create_agent(
    policy_len: int = 1,
    lr_pA: float = 100,
    lr_pB: float = 100,
    learn_A: bool = True,
    noise: float = 0.2,
) -> Agent:
    A_matrix = ego_A(noise=noise)
    B_matrix = ego_B()
    C_matrix = ego_C()
    D_matrix = ego_D()
    pA_matrix = utils.dirichlet_like(A_matrix, scale=1e-32)
    pB_matrix = utils.dirichlet_like(B_matrix, scale=1e-32)
    pD_matrix = utils.dirichlet_like(D_matrix, scale=1e-32)

    agent = Agent(
        A=A_matrix,
        B=B_matrix,
        C=C_matrix,
        D=D_matrix,
        policy_len=policy_len,
        save_belief_hist=True,
        # action_selection="stochastic",
        pA=pA_matrix,
        pB=pB_matrix,
        pD=pD_matrix,
        lr_pA=lr_pA,
        lr_pB=lr_pB,
        lr_pD=lr_pB,
    )
    return agent


def joint_to_observation(my_action: Action, opp_action: Action) -> int:
    if my_action == Action.C and opp_action == Action.C:
        return STATE_CC_IDX
    if my_action == Action.C and opp_action == Action.D:
        return STATE_CD_IDX
    if my_action == Action.D and opp_action == Action.C:
        return STATE_DC_IDX
    return STATE_DD_IDX


class AIFJointAgent(JointWrapper):
    name = "AIF"

    def __init__(
        self,
        policy_len: int = 1,
        lr_pA: float = 1,
        lr_pB: float = 1,
        learn_A: bool = False,
        aware_of_noise: bool = False,
        lr: float | None = None,
    ) -> None:
        super().__init__()
        self.policy_len = policy_len
        # Backward-compat: if `lr` is specified, use it for both A and B
        if lr is not None:
            lr_pA = lr
            lr_pB = lr
        self.lr_pA = lr_pA
        self.lr_pB = lr_pB
        self.learn_A = learn_A
        self.aware_of_noise = aware_of_noise
        self.noise = 0.0
        # noise from match attributes is not available yet; set during first step if requested
        self.agent: Optional[Agent] = None

    def _ensure_agent(self) -> None:
        if self.agent is not None:
            return
        noise = self.match_attributes.get("noise", 0.0) if self.aware_of_noise else 0.0
        self.noise = float(noise)
        self.agent = create_agent(
            policy_len=self.policy_len,
            lr_pA=self.lr_pA,
            lr_pB=self.lr_pB,
            learn_A=self.learn_A,
            noise=self.noise,
        )

    def step(self, state: Tuple[Optional[Action], Optional[Action]]) -> Action:
        self._ensure_agent()
        assert self.agent is not None

        # First decision: no observation yet
        if state[0] is None and state[1] is None:
            self.agent.infer_policies()
            action = self.agent.sample_action()
            assert action.shape[0] == 1, "sample_action should return a single action"
            act_idx = int(action[0])
            return Action.C if act_idx == ACTION_C_IDX else Action.D

        # Convert joint state to observation
        assert state[0] is not None and state[1] is not None
        obs = joint_to_observation(state[0], state[1])

        # State inference and policy selection
        self.agent.infer_states([obs])
        self.agent.infer_policies()
        action = self.agent.sample_action()
        assert action.shape[0] == 1, "sample_action should return a single action"
        act_idx = int(action[0])

        # Optional learning of A and B
        if self.learn_A:
            self.agent.update_A(obs)
        if len(self.agent.qs_hist) > 1:
            self.agent.update_B(qs_prev=self.agent.qs_hist[-2])

        return Action.C if act_idx == ACTION_C_IDX else Action.D

    def reset(self) -> None:  # type: ignore[override]
        if self.agent is not None:
            self.agent.reset()
        super().reset()

class SlowLearnAIF(AIFJointAgent):
    name = "SlowLearnAIF"
    def __init__(self):
        super().__init__(policy_len=1, lr_pA=0.5, lr_pB=0.5, learn_A=True, aware_of_noise=False, lr=None)

class FastLearnAIF(AIFJointAgent):
    name = "FastLearnAIF"
    def __init__(self):
        super().__init__(policy_len=1, lr_pA=1.5, lr_pB=1.5, learn_A=True, aware_of_noise=False, lr=None)

class StdLearnAIF(AIFJointAgent):
    name = "StdLearnAIF"
    def __init__(self):
        super().__init__(policy_len=1, lr_pA=1, lr_pB=1, learn_A=True, aware_of_noise=False, lr=None)

class SlowLearnAIF_P2(AIFJointAgent):
    name = "SlowLearnAIF_P2"
    def __init__(self):
        super().__init__(policy_len=2, lr_pA=0.5, lr_pB=0.5, learn_A=True, aware_of_noise=False, lr=None)

class FastLearnAIF_P2(AIFJointAgent):
    name = "FastLearnAIF_P2"
    def __init__(self):
        super().__init__(policy_len=2, lr_pA=1.5, lr_pB=1.5, learn_A=True, aware_of_noise=False, lr=None)

class StdLearnAIF_P2(AIFJointAgent):
    name = "StdLearnAIF_P2"
    def __init__(self):
        super().__init__(policy_len=2, lr_pA=1, lr_pB=1, learn_A=True, aware_of_noise=False, lr=None)

class SlowLearnAIFNoiseAware(AIFJointAgent):
    name = "SlowLearnAIFNoiseAware"
    def __init__(self):
        super().__init__(policy_len=1, lr_pA=0.5, lr_pB=0.5, learn_A=True, aware_of_noise=True, lr=None)

class FastLearnAIFNoiseAware(AIFJointAgent):
    name = "FastLearnAIFNoiseAware"
    def __init__(self):
        super().__init__(policy_len=1, lr_pA=1.5, lr_pB=1.5, learn_A=True, aware_of_noise=True, lr=None)

class StdLearnAIFNoiseAware(AIFJointAgent):
    name = "StdLearnAIFNoiseAware"
    def __init__(self):
        super().__init__(policy_len=1, lr_pA=1, lr_pB=1, learn_A=True, aware_of_noise=True, lr=None)

class CustomAIF(AIFJointAgent):
    name = "CustomAIF"
    def __init__(self):
        super().__init__(policy_len=2, lr_pA=1, lr_pB=1, learn_A=False, aware_of_noise=True, lr=None)
    
    def _ensure_agent(self) -> None:
        super()._ensure_agent()  # Create the agent first
        if self.agent is not None:
            # Create tit-for-tat B matrix based on the original ego_B() approach
            # B[s_curr, s_next, action] - transition probabilities
            # Start with uniform distribution like the original
            B = np.ones((len(ALL_STATE_IDXS), len(ALL_STATE_IDXS), 2))
            
            # Tit-for-tat strategy: bias transitions based on opponent's last action
            # High probability of choosing action that copies opponent's last move
            
            # From CC state (opponent cooperated last): bias toward cooperation
            B[STATE_CC_IDX, STATE_CC_IDX, ACTION_C_IDX] = 8.0  # high prob: both cooperate
            B[STATE_CC_IDX, STATE_CD_IDX, ACTION_C_IDX] = 2.0  # low prob: opp defects
            B[STATE_CC_IDX, STATE_DC_IDX, ACTION_D_IDX] = 2.0  # low prob: agent defects
            B[STATE_CC_IDX, STATE_DD_IDX, ACTION_D_IDX] = 8.0  # high prob: both defect
            
            # From CD state (opponent defected last): bias toward defection  
            B[STATE_CD_IDX, STATE_DC_IDX, ACTION_D_IDX] = 8.0  # high prob: agent defects, opp cooperates
            B[STATE_CD_IDX, STATE_DD_IDX, ACTION_D_IDX] = 2.0  # low prob: both defect
            B[STATE_CD_IDX, STATE_CC_IDX, ACTION_C_IDX] = 2.0  # low prob: both cooperate
            B[STATE_CD_IDX, STATE_CD_IDX, ACTION_C_IDX] = 8.0  # high prob: agent cooperates, opp defects
            
            # From DC state (opponent cooperated last): bias toward cooperation
            B[STATE_DC_IDX, STATE_CC_IDX, ACTION_C_IDX] = 8.0  # high prob: both cooperate
            B[STATE_DC_IDX, STATE_CD_IDX, ACTION_C_IDX] = 2.0  # low prob: agent cooperates, opp defects
            B[STATE_DC_IDX, STATE_DC_IDX, ACTION_D_IDX] = 2.0  # low prob: agent defects, opp cooperates 
            B[STATE_DC_IDX, STATE_DD_IDX, ACTION_D_IDX] = 8.0  # high prob: both defect
            
            # From DD state (opponent defected last): bias toward defection
            B[STATE_DD_IDX, STATE_DD_IDX, ACTION_D_IDX] = 8.0  # high prob: both defect
            B[STATE_DD_IDX, STATE_DC_IDX, ACTION_D_IDX] = 2.0  # low prob: agent defects, opp cooperates
            B[STATE_DD_IDX, STATE_CD_IDX, ACTION_C_IDX] = 2.0  # low prob: agent cooperates, opp defects
            B[STATE_DD_IDX, STATE_CC_IDX, ACTION_C_IDX] = 8.0  # high prob: both cooperate
            
            # Normalize like the original ego_B() function
            B = B / B.sum(axis=1, keepdims=True)
            self.agent.B = B
   


if __name__ == "__main__":
    from axelrod import Match
    from axelrod.strategies.titfortat import TitForTat

    # Test 1: CustomAIF vs CustomAIF (two tit-for-tat agents)
    print("=== CustomAIF vs CustomAIF ===")
    agent1 = CustomAIF()
    agent2 = CustomAIF()
    match1 = Match((agent1, agent2), turns=1000)
    match1.play()
    print(f"Final scores: {match1.final_score()}")
    print(f"Cooperation counts: {match1.cooperation()}")
    
    # Test 2: CustomAIF vs TitForTat (should behave similarly)
    print("\n=== CustomAIF vs TitForTat ===")
    aif_agent = CustomAIF()
    tft_agent = TitForTat()
    match2 = Match((aif_agent, tft_agent), turns=1000)
    match2.play()
    print(f"Final scores: {match2.final_score()}")
    print(f"Cooperation counts: {match2.cooperation()}")
    
    print("\n=== CustomAIF vs Defector ===")
    aif_agent = CustomAIF()
    tft_agent = axelrod.Defector()
    match2 = Match((aif_agent, tft_agent), turns=1000)
    match2.play()
    print(f"Final scores: {match2.final_score()}")
    print(f"Cooperation counts: {match2.cooperation()}")

    # Test 3: Check specific sequence
    print("\n=== Action sequence (first 10 rounds) ===")
    aif_test = CustomAIF()
    tft_test = TitForTat()
    match3 = Match((aif_test, tft_test), turns=10)
    match3.play()
    print(f"Actions: {[(p1.name, p2.name) for p1, p2 in zip(match3.result[0], match3.result[1])]}")