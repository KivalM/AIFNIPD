import jax.numpy as jnp
import jax.tree_util as jtu
from jax import nn, vmap, lax, jit
from jax import random as jr
import numpy as np
from collections import deque

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
ALL_STATES = [START, CC, CD, DC, DD]
ALL_STATES_LABELS = ["START", "CC", "CD", "DC", "DD"]
ALL_OBS = [START, CC, CD, DC, DD]
ALL_OBS_LABELS = ["START", "CC", "CD", "DC", "DD"]
ALL_ACTIONS = [C, D]
ALL_ACTIONS_LABELS = ["C", "D"]
n_batches = 1
num_states = len(ALL_STATES)
num_obs = len(ALL_OBS)
num_actions = len(ALL_ACTIONS)
import numpy as np

def noisy_obs_matrix(n: float, P0: np.ndarray | None = None) -> np.ndarray:
    """
    Returns P_noisy(o|s) given noise n and a base emission matrix P0(o|s).
    Convention: rows=observations, cols=true states; each column sums to 1.
    If P0 is None, uses the identity (no base noise).
    """
    S = np.array([[1 - n, n],
                  [n, 1 - n]], dtype=float)        # single-player BSC(n)
    K = np.kron(S, S)                               # two players: independent noise
    if P0 is None:
        return K
    return K @ P0                                   # compose channels

def make_agent(
    lr_B: float = 1,
    policy_len: int = 10,
    alpha: float = 1,
    gamma: float = 1,
    bias: float = 0.5,
    preference: str = "standard",
    pB_scale: float = 100,
    noise: float = 0.0,
):
    A_1 = np.eye(num_states)
    P0 = np.eye(num_states-1)
    P_noisy = noisy_obs_matrix(noise, P0)
    A_1[1:,1:] = P_noisy
    assert A_1.sum(axis=1, keepdims=True).all() == 1

    B_1 = np.zeros((num_states, num_states, num_actions))
    # if I cooperate in any state, The next state is CC or CD with equal probability
    B_1[CC, :, C] = bias
    B_1[CD, :, C] = 1 - bias
    # if I defect in any state, The next state is DC or DD with equal probability
    B_1[DC, :, D] = 1 - bias
    B_1[DD, :, D] = bias

    C_1 = np.zeros(num_obs)
    if preference == "standard":
        C_1[CC] = 3
        C_1[CD] = 0
        C_1[DC] = 5
        C_1[DD] = 1
    elif preference == "nash":
        C_1[CC] = 3
        C_1[CD] = 0
        C_1[DC] = 0
        C_1[DD] = 1

    D_1 = np.zeros(num_states)
    D_1[START] = 1

    FINAL_A = [jnp.broadcast_to(A_1, (n_batches,) + (num_obs, num_states))]
    FINAL_B = [jnp.broadcast_to(B_1, (n_batches,) + (num_states, num_states, num_actions))]
    FINAL_C = [jnp.broadcast_to(C_1, (n_batches,) + (num_obs,))]
    FINAL_D = [jnp.broadcast_to(D_1, (n_batches,) + (num_states,))]

    # pb would be a dirichlet distribution of FINAL_B[0] scaled by 10
    pB = jnp.array(utils.dirichlet_like(FINAL_B[0], scale=pB_scale)[0])
    pB = [jnp.broadcast_to(pB, (n_batches,) + (num_states, num_states, num_actions))]

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
        use_param_info_gain=True,
        gamma=jnp.ones(1) * gamma,
        alpha=jnp.ones(1) * alpha,
        onehot_obs=False,
        action_selection="stochastic",
        inference_algo="ovf",
        num_iter=1,
        learn_A=False,
        learn_B=True,
        learn_D=False,
        sampling_mode="marginal",

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


class JaxFiveStateAgentNoisy(JointWrapper):
    name = "JAX_AIF_NOISE"

    def __init__(
        self,
        policy_len: int = 15,
        update_interval: int = 50,
        seed: int = 0,
        lr_B: float = 1,
        alpha: float = 1,
        gamma: float = 1,
        bias: float = 0.5,
        preference: str = "standard",
        pB_scale: float = 100,
    ) -> None:
        super().__init__()
        self.policy_len = policy_len
        self.update_interval = update_interval
        self.seed = seed
        # Use deques with appropriate maxlen for rolling window
        # For N transitions: need N+1 beliefs/obs, N+1 actions (we slice off last)
        self.qs = deque(maxlen=self.update_interval + 1)
        self.obs = deque(maxlen=self.update_interval + 1)
        self.actions = deque(maxlen=self.update_interval + 1)
        self.qpi_history = []  # Store policy distributions
        self.efe_history = []  # Store EFE values
        self.lr_B = lr_B
        self.alpha = alpha
        self.gamma = gamma
        self.pB_scale = pB_scale
        self.bias = bias
        self.preference = preference
        self.step_count = 0  # Track steps since last update
        self.agent = self.make_agent()
        self.empirical_prior = self.agent.D
        self.set_seed(seed)
    
    def make_agent(self):
        return make_agent(policy_len=self.policy_len, lr_B=self.lr_B, alpha=self.alpha, gamma=self.gamma, noise=self.noise, pB_scale=self.pB_scale, bias=self.bias, preference=self.preference)
    
    def receive_match_attributes(self) -> None:  # type: ignore[override]
        super().receive_match_attributes()
        # check if we've initialized the agent already
        if hasattr(self, 'agent') and self.agent is not None:
            self.agent = self.make_agent()

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
        
        # Append in original order: qs, obs, actions together
        # action[t] is taken from qs[t], and results in obs[t+1], qs[t+1] 
        self.qs.append(qs)
        self.obs.append(obs_idx)
        self.actions.append(action)
        self.step_count += 1
        
        # Store EFE from infer_policies (compute it here since step() doesn't return it)
        # qpi, efe = self.agent.infer_policies(qs)
        # self.qpi_history.append(qpi)
        # self.efe_history.append(efe)
        
        # Update every N steps once we have enough data
        # We need N+1 observations/beliefs but only N actions (exclude the last action that hasn't been executed yet)
        if self.step_count >= self.update_interval and len(self.actions) >= self.update_interval and len(self.qs) >= self.update_interval + 1:
            # Convert deques to arrays
            qs_arr = jnp.array([q for q in self.qs])
            obs_arr = jnp.array([o for o in self.obs])
            act_arr = jnp.array([a for a in self.actions])
            
            # Reshape for batch processing
            beliefs = [qs_arr.reshape(n_batches, len(self.qs), num_states)]
            outcomes = obs_arr.reshape(n_batches, len(self.obs))
            # Exclude last action - we haven't seen its result yet
            actions = act_arr[:-1].reshape(n_batches, len(self.actions) - 1, 1)
            
            self.agent = update_B(self.agent, beliefs, outcomes, actions, lr_B=self.lr_B)
            
            self.qpi_history = []
            self.efe_history = []
            self.step_count = 0  # Reset counter after update
        
        return Action.C if action[0][0] == C else Action.D

    def reset(self) -> None:  
        self.qs = deque(maxlen=self.update_interval + 1)
        self.obs = deque(maxlen=self.update_interval + 1)
        self.actions = deque(maxlen=self.update_interval + 1)
        self.qpi_history = []
        self.efe_history = []
        self.step_count = 0
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

    agent = JaxFiveStateAgentNoisy(seed=1, lr_B=1.5, update_interval=100, alpha=1, pB_scale=100, bias=0.5, preference="standard")
    agent2 = axl.DBS()
    agent3 = axl.APavlov2011()

    gtft = axl.TitForTat()
    match1 = Match((agent, gtft), turns=1000, noise=0.15)
    match1.play()
    print(match1.final_score())

    match2 = Match((agent2, gtft), turns=1000, noise=0.15)
    match2.play()
    print(match2.final_score())

    match3 = Match((agent3, gtft), turns=1000, noise=0.15)
    match3.play()
    print(match3.final_score())