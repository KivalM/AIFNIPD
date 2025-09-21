from typing import Tuple
from pymdp.jax.agent import Agent as AIFAgent
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import nn, vmap, lax, jit
from jax import random as jr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pymdp.jax.task import PyMDPEnv
from pymdp.envs import GridWorldEnv
import jax
"""
This agent will be a joint state agent with a joint state space of 5 States:
 - CC: Cooperate, Cooperate
 - CD: Cooperate, Defect
 - DC: Defect, Cooperate
 - DD: Defect, Defect

"""
NUM_STATES = 4
NUM_OBS = 4
NUM_ACTIONS = 2

C = 0
D = 1
CC = 0
CD = 1
DC = 2
DD = 3

ALL_STATE_IDXS = [CC, CD, DC, DD]
ALL_OBS_IDXS = [CC, CD, DC, DD]
ALL_ACTION_IDXS = [C, D]

@jax.jit
def make_A_matrix(noise: float = 0.0, make_uniform: bool = False):
    def make_A_unform():
        A = jnp.ones((len(ALL_STATE_IDXS), len(ALL_OBS_IDXS))) 
        A = A / A.sum(axis=1, keepdims=True)
        return A
    
    def make_A_eye():
        A = jnp.eye(len(ALL_STATE_IDXS), len(ALL_OBS_IDXS)) 
        return A
    return lax.cond(make_uniform, make_A_unform, make_A_eye)

@jax.jit
def make_B_matrix():
    B = jnp.ones((len(ALL_STATE_IDXS), len(ALL_STATE_IDXS), len(ALL_ACTION_IDXS))) 
    B = B / B.sum(axis=1, keepdims=True)
    return B

@jax.jit
def make_C_matrix():
    """
    IPD utilities from the agent's perspective over joint outcomes:
    Index mapping: 0->CC, 1->CD, 2->DC, 3->DD
    Utilities: R=3 (CC), S=0 (CD), T=5 (DC), P=1 (DD)
    """
    return jnp.array([3.0, 0.0, 5.0, 1.0])


@jax.jit
def make_D_matrix():
    D = jnp.ones((len(ALL_STATE_IDXS))) 
    D = D / D.sum(axis=0, keepdims=True)
    return D

@jax.jit
def make_pA_matrix(A_matrix):
    pA = jnp.ones_like(A_matrix)
    pA = pA / pA.sum(axis=1, keepdims=True)
    return pA

@jax.jit
def make_pB_matrix(B_matrix):
    pB = jnp.ones_like(B_matrix)
    pB = pB / pB.sum(axis=1, keepdims=True)
    return pB

@jax.jit
def make_pC_matrix(C_matrix):
    pC = jnp.ones_like(C_matrix)
    pC = pC / pC.sum(axis=1, keepdims=True)
    return pC

@jax.jit
def make_pD_matrix(D_matrix):
    pD = jnp.ones_like(D_matrix)
    pD = pD / pD.sum(axis=1, keepdims=True)
    return pD

def make_agent(
    policy_len: int = 1,
    learn_A: bool = True,
    learn_B: bool = True,
    noise: float = 0.0,
):
    A_matrix = make_A_matrix(noise=noise, make_uniform=learn_A)
    A_matrix = [jnp.broadcast_to(A_matrix, (1, len(ALL_STATE_IDXS), len(ALL_OBS_IDXS)))]
    print(A_matrix[0].shape)
    B_matrix = make_B_matrix()
    B_matrix = [jnp.broadcast_to(B_matrix, (1, len(ALL_STATE_IDXS), len(ALL_STATE_IDXS), len(ALL_ACTION_IDXS)))]
    print(B_matrix[0].shape)
    C_matrix = make_C_matrix()
    C_matrix = [jnp.broadcast_to(C_matrix, (1, len(ALL_OBS_IDXS)))]
    print(C_matrix[0].shape)
    D_matrix = make_D_matrix()
    D_matrix = [jnp.broadcast_to(D_matrix, (1, len(ALL_STATE_IDXS)))]
    print(D_matrix[0].shape)
    pA_matrix = make_pA_matrix(A_matrix[0])
    pA_matrix = [pA_matrix]
    print(pA_matrix[0].shape)
    pB_matrix = make_pB_matrix(B_matrix[0])
    pB_matrix = [pB_matrix]
    print(pB_matrix[0].shape)
    agent = AIFAgent(
        A = A_matrix, 
        B = B_matrix, 
        C = C_matrix, 
        D = D_matrix, 
        pA= None, 
        pB= pB_matrix, 
        E=None, 
        learn_A=False, 
        learn_B=learn_B, 
        learn_D=False, 
        use_utility=True, 
        policy_len=policy_len,
    )
    
    return agent

@jax.jit 
def action_pair_to_obs(action_pair: Tuple[int, int]) -> int:
    def case_I_cooperate(their_action: int):
        return lax.cond(their_action == C, CC, CD)
    def case_I_defect(their_action: int):
        return lax.cond(their_action == C, DC, DD)
    return lax.cond(action_pair[0] == C, case_I_cooperate, case_I_defect, action_pair[1])

@jax.jit 
def obs_to_action_pair(obs: int) -> Tuple[int, int]:
    def case_CC():
        return C, C
    def case_CD():
        return C, D
    def case_DC():
        return D, C
    def case_DD():
        return D, D
    return lax.switch(obs, [case_CC, case_CD, case_DC, case_DD])

@jax.jit
def titfortat(prev_state_idx: int) -> int:
    their_action, _ = obs_to_action_pair(prev_state_idx)
    return their_action

OPP_COOPERATOR = 0
OPP_DEFECTOR = 1
OPP_TFT = 2

@jax.jit
def opponent_action(opponent_type: int, prev_obs: int) -> int:
    def coop():
        return jnp.array(C)
    def defect():
        return jnp.array(D)
    def tft():
        # First step will handle bootstrap externally; here use TFT given prev_obs
        my_prev, their_prev = obs_to_action_pair(prev_obs)
        return my_prev
    return lax.switch(opponent_type, [coop, defect, tft])

@jax.jit
def apply_noise_to_action(rng_key, action: int, noise: float) -> Tuple[int, jax.Array]:
    def flip(a):
        return jnp.where(a == C, D, C)
    rand = jr.uniform(rng_key)
    a_noisy = lax.cond(rand < noise, flip, lambda x: x, action)
    return a_noisy, rand

@jax.jit
def joint_obs_from_actions(my_action: int, opp_action: int) -> int:
    return action_pair_to_obs((my_action, opp_action))

def _initial_uncertain_qs(num_states: int):
    return [jnp.ones((1, 1, num_states)) / num_states]

@jax.jit
def rollout(rng_key, agent, num_timesteps: int, opponent_type: int, noise: float = 0.0):
    """
    Roll out an episode against a fixed opponent type.

    Returns
    - last: carry after final timestep
    - history: dict with keys 'qs', 'action', 'observation'
    Shapes follow the minimal example in scripts/jax.ipynb
    """

    def step_fn(carry, _):
        rng_key = carry["rng_key"]
        ts = carry["ts"]

        def first_step_fn(carry):
            # Prior uncertainty at t=0
            uncertain_qs = _initial_uncertain_qs(NUM_STATES)
            qpi, _ = agent.infer_policies(uncertain_qs)
            # sample my action
            a = agent.sample_action(qpi)
            # empirical prior for next step
            empirical_prior, qs = agent.update_empirical_prior(a, uncertain_qs)

            # opponent's first action: coop for TFT by convention
            opp_a0 = jnp.array(C)
            obs_val = joint_obs_from_actions(a[0][0], opp_a0)
            observation = [jnp.broadcast_to(obs_val, (1, 1))]

            return uncertain_qs, a, observation, empirical_prior

        def subsequent_step_fn(carry):
            observation = carry["observation"]
            prev_action = carry["action"]
            empirical_prior = carry["empirical_prior"]
            prev_obs_scalar = observation[0][0, 0]

            # infer state and choose action
            qs = agent.infer_states(observation, empirical_prior)
            qpi, _ = agent.infer_policies(qs)
            a = agent.sample_action(qpi)

            # opponent responds based on previous joint observation
            opp_a = opponent_action(opponent_type, prev_obs_scalar)

            # optional action noise (NIPD): flip my action with prob=noise
            rng_key_noise, rng_key_next = jr.split(rng_key)
            a_noisy, _ = apply_noise_to_action(rng_key_noise, a[0][0], noise)
            # Build new observation with potentially noisy my action
            obs_val = joint_obs_from_actions(a_noisy, opp_a)
            new_observation = [jnp.broadcast_to(obs_val, (1, 1))]

            new_empirical_prior, _ = agent.update_empirical_prior(a, qs)
            return qs, a, new_observation, new_empirical_prior

        qs, action, observation, empirical_prior = lax.cond(ts == 0, first_step_fn, subsequent_step_fn, carry)

        # record executed action (for t=0 equals planned since no noise applied then)
        executed = lax.cond(
            ts == 0,
            lambda _: action[0],
            lambda _: jnp.expand_dims(jnp.array([observation[0][0, 0] // 2], dtype=jnp.int32), 0),
            operand=None,
        )

        history_slice = {
            "qs": qs[0],
            "action": action[0],
            "action_executed": executed,
            "observation": observation[0],
        }

        new_carry = {
            "rng_key": jr.split(rng_key, 1)[0],
            "action": action,
            "observation": observation,
            "empirical_prior": empirical_prior,
            "ts": ts + 1,
        }
        return new_carry, history_slice

    initial_carry = {
        "rng_key": jr.split(rng_key, 1)[0],
        "action": jnp.zeros((1, 1), dtype=jnp.int32),
        "observation": [jnp.zeros((1, 1), dtype=jnp.int32)],
        "empirical_prior": [jnp.zeros((1, NUM_STATES), dtype=jnp.float32)],
        "ts": 0,
    }

    last, history = lax.scan(step_fn, initial_carry, None, length=num_timesteps + 1)
    return last, history

def _history_to_beliefs_actions(history):
    """
    Convert scan history into (qs_time_series, obs_time_series, actions_time_series)
    Shapes will match agent.infer_parameters expectations as per scripts/jax.ipynb
    """
    # prefer executed actions if present
    actions = history.get("action_executed", history["action"])[1:]  # drop t=0 bootstrap
    actions = jnp.broadcast_to(actions, (1, actions.shape[0], actions.shape[1]))  # (1, T, 1)

    observations = history["observation"]  # (T+1, 1, 1)
    obs_list = []
    for observation in observations:
        obs_list.append(observation[0, 0])
    obs = jnp.array([obs_list])  # (1, T+1)

    qs_hist = history["qs"]  # list length T+1, each (1,1,NUM_STATES)
    qs_list = []
    for q in qs_hist:
        q = q.squeeze(0).squeeze(0)
        qs_list.append(q)
    qs = jnp.array(qs_list)
    qs = jnp.broadcast_to(qs, (1, qs.shape[0], qs.shape[1]))  # (1, T+1, NUM_STATES)
    return [qs], obs, actions

def train_against_opponent(rng_key, agent, opponent_type: int, episode_length: int = 50, lr_pA: float = 1.0, lr_pB: float = 1.0, noise: float = 0.0):
    """
    Run one training episode against an opponent and update agent parameters (A/B) using infer_parameters.
    Returns the updated agent and rollout history.
    """
    jitted_rollout = jit(rollout, static_argnums=(2, 3,))
    last, history = jitted_rollout(rng_key, agent, episode_length, opponent_type, noise)
    beliefs_A, obs, actions = _history_to_beliefs_actions(history)
    updated_agent = agent.infer_parameters(beliefs_A, obs, actions, lr_pA=lr_pA, lr_pB=lr_pB)
    return updated_agent, history

def train_across_opponents(rng_key, agent, num_episodes_per_opponent: int = 10, episode_length: int = 50, lr_pA: float = 1.0, lr_pB: float = 1.0, noise: float = 0.0):
    """
    Cycle training across three opponent types: cooperator, defector, tit-for-tat.
    Returns the trained agent.
    """
    opponent_seq = [OPP_COOPERATOR, OPP_DEFECTOR, OPP_TFT]
    key = rng_key
    for opp in opponent_seq:
        for _ in range(num_episodes_per_opponent):
            key, sub = jr.split(key)
            agent, _ = train_against_opponent(sub, agent, opp, episode_length=episode_length, lr_pA=lr_pA, lr_pB=lr_pB, noise=noise)
    return agent

if __name__ == "__main__":
    key = jr.PRNGKey(0)
    agent = make_agent(policy_len=3, learn_A=False, learn_B=True)
    print(agent.batch_size)
    # quick smoke test
    agent = train_across_opponents(key, agent, num_episodes_per_opponent=1, episode_length=10, noise=0.0)
