from typing import Optional, Tuple
from axelrod import Action
import numpy as np
from pymdp.agent import Agent
import pymdp.utils as utils
import itertools
from agents.wrapper import JointWrapper

# Constants for actions
C = 0  # Cooperate
D = 1  # Defect
ALL_ACTIONS = [C, D]

# Hidden state factors (two factors: ego action, alter action)
ALL_EGO_ACTIONS = [C, D]      # Your previous action
ALL_ALTER_ACTIONS = [C, D]    # Opponent's previous action

# Observations: joint action outcome (same as before)
CC = 0
CD = 1
DC = 2
DD = 3
ALL_OBSERVATIONS = [CC, CD, DC, DD]

def factorized_A(noise=0.2):
    # A: observation likelihood P(o | s1, s2) where s1=ego action, s2=alter action
    A = utils.obj_array(1)
    A[0] = np.full((len(ALL_OBSERVATIONS), len(ALL_EGO_ACTIONS), len(ALL_ALTER_ACTIONS)), noise / 3)
    A[0][CC, C, C] = 1 - noise
    A[0][CD, C, D] = 1 - noise
    A[0][DC, D, C] = 1 - noise
    A[0][DD, D, D] = 1 - noise
    return A

def factorized_B():
    # B: identity transitions (no prior assumption)
    # B_ego = np.ones((len(ALL_EGO_ACTIONS), len(ALL_EGO_ACTIONS), len(ALL_ACTIONS)))
    B_ego = np.ones((len(ALL_EGO_ACTIONS), len(ALL_EGO_ACTIONS), len(ALL_ACTIONS)))
    B_ego = B_ego / B_ego.sum(axis=0)
    
    B_alter = np.ones((len(ALL_ALTER_ACTIONS), len(ALL_ALTER_ACTIONS), len(ALL_ACTIONS)))
    B_alter = B_alter / B_alter.sum(axis=0)

    B = utils.obj_array(2)
    B[0] = B_ego
    B[1] = B_alter
    return B

def factorized_C():
    # Preference over observations
    C = np.zeros(len(ALL_OBSERVATIONS))
    C[CC] = 5
    C[CD] = 0
    C[DC] = 3
    C[DD] = 1
    return C

def factorized_D():
    D_ego = np.array([1, 0])
    D_alter = np.array([1, 0])
    D = utils.obj_array(2)
    D[0] = D_ego
    D[1] = D_alter
    return D

def create_policies(policy_len=1):
    cooperate = [0, 0]
    defect = [1, 1]
    one_step_actions = [cooperate, defect]
    all_policy_sequences = list(itertools.product(one_step_actions, repeat=policy_len))
    policies = [np.array(seq) for seq in all_policy_sequences]
    return policies

def create_agent_factorized(
    policy_len: int = 1,
    lr_pA: float = 100,
    lr_pB: float = 100,
    learn_A: bool = True,
    noise: float = 0.2,
    policies  = None,
    B_matrix: np.ndarray | None = None,
    p_B_matrix: np.ndarray | None = None,
):
    A_matrix = factorized_A(noise=noise)
    B_matrices = factorized_B()
    C_matrix = factorized_C()
    D_matrices = factorized_D()
    if policies is None:
        policies = create_policies(policy_len)
 
    B_matrices = factorized_B() if B_matrix is None else B_matrix
    
    pA_matrix = utils.dirichlet_like(A_matrix, scale=1e-32)
    pB_matrices = utils.dirichlet_like(B_matrices, scale=1e-32) if p_B_matrix is None else p_B_matrix
    pD_matrices = utils.dirichlet_like(D_matrices, scale=1e-32)

    agent = Agent(
        A=A_matrix,
        B=B_matrices,
        C=C_matrix,
        D=D_matrices,
        # policies=policies,
        pA=pA_matrix,
        pB=pB_matrices,
        pD=pD_matrices,
        # E=np.array([1 for _ in range(len(policies))]),
        lr_pA=lr_pA,
        lr_pB=lr_pB,
        lr_pD=lr_pB,
        # policy_len=policy_len,
        save_belief_hist=True,
        # action_selection="stochastic"
        sampling_mode="full",
        control_fac_idx=[0]
    )
    return agent

def joint_to_observation(my_action: Action, opponent_action: Action) -> int:
    if my_action == Action.C and opponent_action == Action.C:
        return CC
    if my_action == Action.C and opponent_action == Action.D:
        return CD
    if my_action == Action.D and opponent_action == Action.C:
        return DC
    return DD


def _extract_vector_action(sampled) -> int:
    arr = np.array(sampled)
    # assert np.array_equal(arr, np.array([1, 1])) or np.array_equal(arr, np.array([0, 0])), "sample_action should return vector"
    return int(arr[1])


class Factorized(JointWrapper):
    name = "Factorized AIF"

    def __init__(
        self,
        policy_len: int = 1,
        lr_pA: float = 1,
        lr_pB: float = 1,
        learn_A: bool = False,
        aware_of_noise: bool = False,
        B_matrix: np.ndarray | None = None,
        p_B_matrix: np.ndarray | None = None,
    ):
        super().__init__()
        self.policy_len = policy_len
        self.learn_A = learn_A
        self.aware_of_noise = aware_of_noise
        self.lr_pA = lr_pA
        self.lr_pB = lr_pB
        self.noise = 0.0
        self.policies = create_policies(policy_len)
        self.B_matrix = B_matrix
        self.p_B_matrix = p_B_matrix
        self.agent = create_agent_factorized(
            policy_len=self.policy_len,
            lr_pA=self.lr_pA,
            lr_pB=self.lr_pB,
            learn_A=self.learn_A,
            noise=self.noise,
            B_matrix=self.B_matrix,
            p_B_matrix=self.p_B_matrix,
        )

    def step(self, state: Tuple[Optional[Action], Optional[Action]]) -> Action:
        if state[0] is None and state[1] is None:
            self.agent.infer_policies()
            sampled = self.agent.sample_action()
            act_idx = _extract_vector_action(sampled)
            return Action.C if act_idx == C else Action.D

        # Convert joint state to observation index
        assert state[0] is not None and state[1] is not None
        observation = joint_to_observation(state[0], state[1])
        self.agent.infer_states([observation])
        self.agent.infer_policies()
        sampled = self.agent.sample_action()
        act_idx = _extract_vector_action(sampled)

        if self.learn_A:
            self.agent.update_A(observation)
        if len(self.agent.qs_hist) > 1:
            self.agent.update_B(qs_prev=self.agent.qs_hist[-2])
            # decay prior probabilities of actions
            # self.agent.pB[0] = self.agent.pB[0] * 0.95  
            # self.agent.pB[1] = self.agent.pB[1] * 0.95
        return Action.C if act_idx == C else Action.D

    def reset(self):
        if self.agent is not None:
            self.agent.reset()
        super().reset()



if __name__ == "__main__":
    from axelrod import Match
    from axelrod.strategies.titfortat import TitForTat
    import axelrod
    # Test 1: CustomAIF vs CustomAIF (two tit-for-tat agents)
    print("=== CustomAIF vs CustomAIF ===")
    agent1 = Factorized(5)
    agent2 = Factorized(5)
    match1 = Match((agent1, agent2), turns=1000)
    match1.play()
    print(f"Final scores: {match1.final_score()}")
    print(f"Cooperation counts: {match1.cooperation()}")
    
    # Test 2: CustomAIF vs TitForTat (should behave similarly)
    print("\n=== CustomAIF vs TitForTat ===")
    aif_agent = Factorized()
    tft_agent = TitForTat()
    match2 = Match((aif_agent, tft_agent), turns=1000)
    match2.play()
    print(f"Final scores: {match2.final_score()}")
    print(f"Cooperation counts: {match2.cooperation()}")
    
    print("\n=== CustomAIF vs Defector ===")
    aif_agent = Factorized()
    tft_agent = axelrod.Defector()
    match2 = Match((aif_agent, tft_agent), turns=1000)
    match2.play()
    print(f"Final scores: {match2.final_score()}")
    print(f"Cooperation counts: {match2.cooperation()}")

    # Test 3: Check specific sequence
    print("\n=== Action sequence (first 10 rounds) ===")
    aif_test = Factorized()
    tft_test = TitForTat()
    match3 = Match((aif_test, tft_test), turns=10)
    match3.play()
    print(f"Actions: {[(p1.name, p2.name) for p1, p2 in zip(match3.result[0], match3.result[1])]}")