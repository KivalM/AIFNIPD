"""Agent configuration UI components."""

import streamlit as st
from typing import Dict, Any, Tuple
import axelrod as axl

# Import agents - handle both absolute and relative imports
try:
    from agents.aif.jax.five_state import JaxFiveStateAgent
    from agents.aif.jax.five_state_noise import JaxFiveStateAgentNoisy
    from agents.aif.jax.five_state_utility import JaxFiveStateAgentUtility
    from agents.qlearner import JaxQLearner, CooperativeQLearner
    from agents.bqlearner import JaxBayesianQLearner, CooperativeBQLearner
    from agents.psrl import PSRL, CooperativePSRL
    from agents.dynaQ import DynaQ, CooperativeDynaQ
except ImportError:
    # Fallback if running as module
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(src_path))
    from agents.aif.jax.five_state import JaxFiveStateAgent
    from agents.aif.jax.five_state_noise import JaxFiveStateAgentNoisy
    from agents.aif.jax.five_state_utility import JaxFiveStateAgentUtility
    from agents.qlearner import JaxQLearner, CooperativeQLearner
    from agents.bqlearner import JaxBayesianQLearner, CooperativeBQLearner
    from agents.psrl import PSRL, CooperativePSRL
    from agents.dynaQ import DynaQ, CooperativeDynaQ


def configure_agent() -> Tuple[Any, Dict[str, Any]]:
    """Configure agent through UI and return agent instance and parameters."""
    st.sidebar.header("Agent Configuration")
    
    agent_type = st.sidebar.selectbox(
        "Agent Type",
        [
            "AIF (Five State)",
            "AIF (Five State Noisy)",
            "AIF (Five State Utility)",
            "Q-Learning",
            "Cooperative Q-Learning",
            "Bayesian Q-Learning",
            "Cooperative Bayesian Q-Learning",
            "PSRL",
            "Cooperative PSRL",
            "DynaQ",
            "Cooperative DynaQ",
        ]
    )
    
    params = {}
    
    if agent_type.startswith("AIF"):
        st.sidebar.subheader("AIF Parameters")
        params['policy_len'] = st.sidebar.slider("Policy Length", 1, 20, 10)
        params['update_interval'] = st.sidebar.slider("Update Interval", 1, 100, 50)
        params['seed'] = st.sidebar.number_input("Seed", 0, 1000, 42)
        params['lr_B'] = st.sidebar.slider("Learning Rate B", 0.1, 5.0, 1.0, 0.1)
        params['alpha'] = st.sidebar.slider("Alpha", 0.1, 5.0, 1.0, 0.1)
        params['gamma'] = st.sidebar.slider("Gamma", 0.1, 5.0, 1.0, 0.1)
        params['bias'] = st.sidebar.slider("Bias", 0.0, 1.0, 0.5, 0.05)
        params['preference'] = st.sidebar.selectbox("Preference", ["standard", "nash"])
        params['pB_scale'] = st.sidebar.slider("pB Scale", 1, 200, 100)
        
        if agent_type == "AIF (Five State)":
            agent = JaxFiveStateAgent(**params)
        elif agent_type == "AIF (Five State Noisy)":
            agent = JaxFiveStateAgentNoisy(**params)
        elif agent_type == "AIF (Five State Utility)":
            agent = JaxFiveStateAgentUtility(**params)
    
    elif agent_type == "Q-Learning":
        st.sidebar.subheader("Q-Learning Parameters")
        params['learning_rate'] = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.9, 0.01)
        params['discount_rate'] = st.sidebar.slider("Discount Rate", 0.1, 1.0, 0.9, 0.01)
        params['action_selection_parameter'] = st.sidebar.slider("Epsilon", 0.0, 1.0, 0.1, 0.01)
        agent = JaxQLearner(**params)
    
    elif agent_type == "Cooperative Q-Learning":
        st.sidebar.subheader("Cooperative Q-Learning Parameters")
        params['learning_rate'] = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.9, 0.01)
        params['discount_rate'] = st.sidebar.slider("Discount Rate", 0.1, 1.0, 0.9, 0.01)
        params['action_selection_parameter'] = st.sidebar.slider("Epsilon", 0.0, 1.0, 0.1, 0.01)
        agent = CooperativeQLearner(**params)
    
    elif agent_type == "Bayesian Q-Learning":
        st.sidebar.subheader("Bayesian Q-Learning Parameters")
        params['discount_rate'] = st.sidebar.slider("Discount Rate", 0.1, 1.0, 0.5, 0.01)
        params['initial_variance'] = st.sidebar.slider("Initial Variance", 0.1, 10.0, 1.0, 0.1)
        params['reward_variance'] = st.sidebar.slider("Reward Variance", 0.1, 10.0, 1.0, 0.1)
        agent = JaxBayesianQLearner(**params)
    
    elif agent_type == "Cooperative Bayesian Q-Learning":
        st.sidebar.subheader("Cooperative Bayesian Q-Learning Parameters")
        params['discount_rate'] = st.sidebar.slider("Discount Rate", 0.1, 1.0, 0.5, 0.01)
        params['initial_variance'] = st.sidebar.slider("Initial Variance", 0.1, 10.0, 1.0, 0.1)
        params['reward_variance'] = st.sidebar.slider("Reward Variance", 0.1, 10.0, 1.0, 0.1)
        agent = CooperativeBQLearner(**params)
    
    elif agent_type == "PSRL":
        st.sidebar.subheader("PSRL Parameters")
        params['prior_strength'] = st.sidebar.slider("Prior Strength", 0.1, 5.0, 0.5, 0.1)
        params['discount_rate'] = st.sidebar.slider("Discount Rate", 0.1, 1.0, 0.95, 0.01)
        params['value_iteration_steps'] = st.sidebar.slider("Value Iteration Steps", 10, 200, 50)
        agent = PSRL(**params)
    
    elif agent_type == "Cooperative PSRL":
        st.sidebar.subheader("Cooperative PSRL Parameters")
        params['prior_strength'] = st.sidebar.slider("Prior Strength", 0.1, 5.0, 0.5, 0.1)
        params['discount_rate'] = st.sidebar.slider("Discount Rate", 0.1, 1.0, 0.95, 0.01)
        params['value_iteration_steps'] = st.sidebar.slider("Value Iteration Steps", 10, 200, 50)
        agent = CooperativePSRL(**params)
    
    elif agent_type == "DynaQ":
        st.sidebar.subheader("DynaQ Parameters")
        params['learning_rate'] = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
        params['discount_rate'] = st.sidebar.slider("Discount Rate", 0.1, 1.0, 0.9, 0.01)
        params['action_selection_parameter'] = st.sidebar.slider("Epsilon", 0.0, 1.0, 0.1, 0.01)
        params['planning_steps'] = st.sidebar.slider("Planning Steps", 1, 20, 5)
        agent = DynaQ(**params)
    
    elif agent_type == "Cooperative DynaQ":
        st.sidebar.subheader("Cooperative DynaQ Parameters")
        params['learning_rate'] = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
        params['discount_rate'] = st.sidebar.slider("Discount Rate", 0.1, 1.0, 0.9, 0.01)
        params['action_selection_parameter'] = st.sidebar.slider("Epsilon", 0.0, 1.0, 0.1, 0.01)
        params['planning_steps'] = st.sidebar.slider("Planning Steps", 1, 20, 5)
        agent = CooperativeDynaQ(**params)
    
    return agent, params


def configure_opponent() -> Any:
    """Configure opponent through UI and return opponent instance."""
    st.sidebar.header("Opponent Configuration")
    
    opponent_type = st.sidebar.selectbox(
        "Opponent Type",
        [
            "AIF (Five State)",
            "AIF (Five State Noisy)",
            "AIF (Five State Utility)",
            "Q-Learning",
            "Cooperative Q-Learning",
            "Bayesian Q-Learning",
            "Cooperative Bayesian Q-Learning",
            "PSRL",
            "Cooperative PSRL",
            "DynaQ",
            "Cooperative DynaQ",
            "TitForTat",
            "Always Cooperate",
            "Always Defect",
            "Random",
            "Grudger",
            "DBS",
        ]
    )
    
    if opponent_type.startswith("AIF"):
        # Use default parameters for opponent
        if opponent_type == "AIF (Five State)":
            return JaxFiveStateAgent()
        elif opponent_type == "AIF (Five State Noisy)":
            return JaxFiveStateAgentNoisy()
        elif opponent_type == "AIF (Five State Utility)":
            return JaxFiveStateAgentUtility()
    elif opponent_type == "Q-Learning":
        return JaxQLearner()
    elif opponent_type == "Cooperative Q-Learning":
        return CooperativeQLearner()
    elif opponent_type == "Bayesian Q-Learning":
        return JaxBayesianQLearner()
    elif opponent_type == "Cooperative Bayesian Q-Learning":
        return CooperativeBQLearner()
    elif opponent_type == "PSRL":
        return PSRL()
    elif opponent_type == "Cooperative PSRL":
        return CooperativePSRL()
    elif opponent_type == "DynaQ":
        return DynaQ()
    elif opponent_type == "Cooperative DynaQ":
        return CooperativeDynaQ()
    elif opponent_type == "TitForTat":
        return axl.TitForTat()
    elif opponent_type == "Always Cooperate":
        return axl.Cooperator()
    elif opponent_type == "Always Defect":
        return axl.Defector()
    elif opponent_type == "Random":
        return axl.Random()
    elif opponent_type == "Grudger":
        return axl.Grudger()
    elif opponent_type == "DBS":
        return axl.DBS()
    
    return axl.TitForTat()  # Default


def configure_game() -> Dict[str, Any]:
    """Configure game parameters."""
    st.sidebar.header("Game Configuration")
    
    turns = st.sidebar.slider("Number of Turns", 10, 1000, 100)
    noise = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.0, 0.01)
    update_interval = st.sidebar.slider("Visualization Update Interval", 1, 50, 1)
    
    return {
        'turns': turns,
        'noise': noise,
        'update_interval': update_interval,
    }

