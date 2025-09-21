import jax.numpy as jnp
import jax.tree_util as jtu
from jax import nn, vmap, lax, jit
from jax import random as jr
import numpy as np

from pymdp.envs import GridWorldEnv
from pymdp.jax.task import PyMDPEnv
from pymdp.jax.agent import Agent as AIFAgent

import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional, Tuple

import axelrod
import numpy as np
from axelrod.action import Action
from pymdp.agent import Agent
import pymdp.utils as utils
from ...wrapper import C, JointWrapper

START = 0
CC = 1
CD = 2
DC = 3
DD = 4

C = 0
D = 1
ALL_STATES = [START, C, D]
ALL_STATES_LABELS = ["START", "C", "D"]
ALL_OBS = [START, CC, CD, DC, DD]
ALL_OBS_LABELS = ["START", "CC", "CD", "DC", "DD"]
ALL_ACTIONS = [C, D]
ALL_ACTIONS_LABELS = ["C", "D"]
n_batches = 1
num_states = len(ALL_STATES)
num_obs = len(ALL_OBS)
num_actions = len(ALL_ACTIONS)

def make_agent(
    lr_B: float = 1,
    policy_len: int = 10,
    alpha: float = 1,
    gamma: float = 1,
    bias: float = 0.5,
    preference: str = "standard",
    noisy_A: bool = False,
    noise: float = 0.0,
    pB_scale: float = 100,
):

    
    
    if noisy_A:
        def obs_tensor_via_kron(n: float) -> np.ndarray:
            S = np.array([[1 - n, n],
                        [n, 1 - n]], dtype=float)
            K = np.kron(S, S)                  # 4x4: P(obs_pair | state_pair)
            return K.reshape(4, 2, 2)          # [obs, s1, s2] with obs=[CC,CD,DC,DD]
        non_start_part = obs_tensor_via_kron(noise)
        A_1 = np.zeros((num_obs, num_states, num_states))
        A_1[START, START, START] = 1
        A_1[1:, 1:, 1:] = non_start_part
    else:
        A_1 = np.zeros((num_obs, num_states, num_states))
        A_1[START, START, START] = 1
        A_1[CC, C, C] = 1
        A_1[CD, C, D] = 1
        A_1[DC, D, C] = 1
        A_1[DD, D, D] = 1
    
    B_1 = np.zeros((num_states, num_states, num_actions))
    B_1[C, :, C] = 1
    B_1[D, :, D] = 1
    B_2 = np.zeros((num_states, num_states, 1))
    # if I cooperate in any state, The next state is CC or CD with equal probability
    B_2[C, :, 0] = bias
    B_2[D, :, 0] = 1 - bias


    C_1 = np.zeros(num_obs)
    if preference == "standard":    
        C_1[CC] = 3
        C_1[CD] = 0
        C_1[DC] = 4
        C_1[DD] = 1
    elif preference == "nash":
        C_1[CC] = 3
        C_1[CD] = 0
        C_1[DC] = 0
        C_1[DD] = 1

    D_1 = np.zeros(num_states)
    D_2 = np.zeros(num_states)
    D_1[START] = 1
    D_2[START] = 1

    FINAL_A = [jnp.broadcast_to(A_1, (n_batches,) + (num_obs, num_states, num_states))]
    FINAL_B = [jnp.broadcast_to(B_1, (n_batches,) + (num_states, num_states, num_actions)), jnp.broadcast_to(B_2, (n_batches,) + (num_states, num_states, 1))]
    FINAL_C = [jnp.broadcast_to(C_1, (n_batches,) + (num_obs,))]
    FINAL_D = [jnp.broadcast_to(D_1, (n_batches,) + (num_states,)), jnp.broadcast_to(D_2, (n_batches,) + (num_states,))]

    pA = [jnp.ones_like(FINAL_A[0]) / num_obs]

    pB_0 = utils.dirichlet_like(FINAL_B[0], scale=pB_scale)[0].reshape(FINAL_B[0].shape)
    pB_1 = utils.dirichlet_like(FINAL_B[1], scale=pB_scale)[0].reshape(FINAL_B[1].shape)
    pB = [jnp.broadcast_to(pB_0, FINAL_B[0].shape), jnp.broadcast_to(pB_1, FINAL_B[1].shape)]

    agent = AIFAgent(
        FINAL_A,
        FINAL_B,
        FINAL_C,
        FINAL_D,
        E=None,
        pA=None,
        pB=pB,
        policy_len=policy_len,
        use_utility=True,
        use_states_info_gain=True,
        use_param_info_gain=False,
        gamma=jnp.ones(1) * gamma,
        alpha=jnp.ones(1) * alpha,
        onehot_obs=False,
        action_selection="stochastic",
        inference_algo="ovf",
        num_iter=16,
        learn_A=False,
        learn_B=True,
        learn_D=False,
    )

    return agent

def action_pair_to_obs(action, opponent_action):
    own_action = action
    opponent_action = opponent_action
    # CC -> 1 , CD -> 2, DC -> 3, DD -> 4
    # get idx from pair
    idx = 1 +(own_action * 2) + opponent_action
    return jnp.array([[idx]])

@jit
def step(rng_key, agent, obs_idx, empirical_prior):
    qs = agent.infer_states(obs_idx, empirical_prior)
    qpi, _ = agent.infer_policies(qs)
    rng_key = jr.split(rng_key)
    action = agent.sample_action(qpi, rng_key=rng_key[1:])
    empirical_prior, qs = agent.update_empirical_prior(action, qs)
    return rng_key[0], obs_idx, empirical_prior, qs, action


@jit
def update_B(agent, beliefs, outcomes, actions, lr_B=1):
    return agent.infer_parameters(beliefs, outcomes, actions, lr_B=lr_B)


class JaxFactorizedAgent(JointWrapper):
    name = "JAX_AIF_FACTORIZED"

    def __init__(
        self,
        policy_len: int = 1,
        update_interval: int = 50,
        seed: int = 0,
        lr_B: float = 1,
        alpha: float = 1,
        gamma: float = 1,
        noisy_A: bool = False,
        bias: float = 0.5,
        preference: str = "standard",
        pB_scale: float = 100,
    ) -> None:
        super().__init__()
        self.policy_len = policy_len
        self.agent = make_agent(policy_len=policy_len, lr_B=lr_B, alpha=alpha, gamma=gamma, noisy_A=noisy_A, bias=bias, preference=preference, pB_scale=pB_scale)
        self.update_interval = update_interval
        self.empirical_prior = self.agent.D
        self.seed = seed
        self.qs = []
        self.obs = []
        self.actions = []
        self.lr_B = lr_B
        self.noisy_A = noisy_A
        self.bias = bias
        self.preference = preference
        self.pB_scale = pB_scale
        self.set_seed(seed)
    
    def set_seed(self, seed: int):
        self.seed = seed
        self.rng_key = jr.PRNGKey(self.seed)
        self.rng_key = jr.split(self.rng_key)[1]

    def step(self, state: Tuple[Optional[Action], Optional[Action]]) -> Action:
        obs_idx = None
        if state[0] is None and state[1] is None:
            obs_idx = [jnp.broadcast_to(jnp.array([START]), (1,1))]
        else:
            obs_idx = [action_pair_to_obs(state[0].value, state[1].value)]
        
        self.rng_key = jr.split(self.rng_key)[1]
        rng_key, obs_idx, empirical_prior, qs, action = step(self.rng_key, self.agent, obs_idx, self.empirical_prior)
        self.empirical_prior = empirical_prior
        self.qs.append(qs)
        self.obs.append(obs_idx)
        self.actions.append(action)
        if len(self.obs) % self.update_interval == 0:
          qs = np.array(self.qs)    
          qs = qs.reshape(len(self.qs), 2, num_states)
          beliefs = [jnp.array(qs[:, 0, :]).reshape(n_batches, len(self.qs), num_states), jnp.array(qs[:, 1, :]).reshape(n_batches, len(self.qs), num_states)]
          actions = [jnp.array(self.actions).reshape(n_batches, len(self.actions),2)[:, :-1, :]]
          outcomes = [jnp.array(self.obs).reshape(n_batches, len(self.obs))]
          self.agent = update_B(self.agent, beliefs, outcomes[0], actions[0], lr_B=self.lr_B)
          self.qs = []
          self.obs = []
          self.actions = []
        return Action.C if action[0][0] == C else Action.D

    def reset(self) -> None:  
        self.qs = []
        self.obs = []
        self.actions = []
        self.empirical_prior = self.agent.D
        super().reset()


### Utility Functions
def plot_A_1(A_1):
    plt.figure(figsize=(4, 4))
    sns.heatmap(A_1, annot=True, cmap='viridis', cbar=False)
    # label axes 
    ticks = [i + 0.5 for i in range(num_states)]
    plt.xticks(ticks=ticks, labels=ALL_STATES_LABELS)
    plt.yticks(ticks=ticks, labels=ALL_STATES_LABELS)
    
    
    plt.title('Transition Matrix A_1')
    plt.xlabel('Observation')
    plt.ylabel('State')
    plt.show()

def plot_B_1(B_1):
    # B_1 Shape (next_state, current_state, action)
    assert B_1.shape == (num_states, num_states, num_actions), f"B_1 shape is {B_1.shape}, expected (num_states, num_states, num_actions)"
    plt.figure(figsize=(8, 4))
    for action in range(num_actions):
        plt.subplot(1, num_actions, action + 1)
        sns.heatmap(B_1[:, :, action], annot=True, cmap='viridis', cbar=False)
        # offset labels on ticks by 0.5
        ticks = [i + 0.5 for i in range(num_states)]
        plt.xticks(ticks=ticks, labels=ALL_STATES_LABELS)
        plt.yticks(ticks=ticks, labels=ALL_STATES_LABELS)
        plt.title(f'B_1 for action {ALL_ACTIONS_LABELS[action]}')
        plt.xlabel('Next State')
        plt.ylabel('Current State')
        plt.tight_layout()
  
    plt.show()

def plot_C_1(C_1, title="Preferences"):
    plt.grid(zorder=0)
    plt.bar(range(C_1.shape[0]), C_1, color='r', zorder=3)
    plt.xticks(range(C_1.shape[0]), ALL_OBS_LABELS)
    plt.title(title)
    plt.show()



if __name__ == "__main__":
    import axelrod as axl
    from axelrod import Match
    from axelrod.strategies.titfortat import TitForTat
    import jax
    # with jax.disable_jit():
    agent = JaxFactorizedAgent(seed=0, lr_B=1, update_interval=10, alpha=1, noisy_A=False, bias=0.50, preference="nash", pB_scale=1)
    # agent2 = JaxFactorizedAgent(seed=1, lr_B=10, update_interval=10, alpha=0.5)
    agent2 = axl.DBS()
    agent3 = axl.APavlov2011()
    agent4 = axl.WinStayLoseShift()
    
    zde = axl.Gradual()
    # match1 = Match((agent, axl.Defector()), turns=1000, noise=0.00)
    # match1.play()
    # print(match1.final_score())
    match2 = Match((agent, zde), turns=1000, noise=0.05)
    match2.play()
    print(match2.cooperation())
    print(match2.final_score_per_turn())
    
    match3 = Match((agent2, zde), turns=1000, noise=0.05)
    match3.play()
    print(match3.cooperation())
    print(match3.final_score_per_turn())
    
    match4 = Match((agent3, zde), turns=1000, noise=0.05)
    match4.play()
    print(match4.cooperation())
    print(match4.final_score_per_turn())

    match5 = Match((agent4, zde), turns=1000, noise=0.05)
    match5.play()
    print(match5.cooperation())
    print(match5.final_score_per_turn())