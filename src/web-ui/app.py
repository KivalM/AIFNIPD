"""Main Streamlit app for IPD agent interaction."""

import streamlit as st
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Ensure src is in path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Ensure web-ui directory is in path for relative imports
web_ui_path = Path(__file__).parent
if str(web_ui_path) not in sys.path:
    sys.path.insert(0, str(web_ui_path))

# Import using importlib to handle hyphenated directory name
import importlib.util

def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    # Set __package__ to help with relative imports
    module.__package__ = "web_ui"
    spec.loader.exec_module(module)
    return module

agent_config = import_module_from_path("agent_config", web_ui_path / "agent_config.py")
agent_wrapper_mod = import_module_from_path("agent_wrapper", web_ui_path / "agent_wrapper.py")
simulator_mod = import_module_from_path("simulator", web_ui_path / "simulator.py")
matrices_mod = import_module_from_path("matrices", web_ui_path / "visualizers" / "matrices.py")
efe_mod = import_module_from_path("efe", web_ui_path / "visualizers" / "efe.py")
history_mod = import_module_from_path("history", web_ui_path / "components" / "history.py")
stats_mod = import_module_from_path("stats", web_ui_path / "components" / "stats.py")

configure_agent = agent_config.configure_agent
configure_opponent = agent_config.configure_opponent
configure_game = agent_config.configure_game
AIFAgentStateWrapper = agent_wrapper_mod.AIFAgentStateWrapper
simulate_game = simulator_mod.simulate_game
StepByStepSimulator = simulator_mod.StepByStepSimulator
visualize_A_matrix = matrices_mod.visualize_A_matrix
visualize_B_matrix = matrices_mod.visualize_B_matrix
visualize_C_matrix = matrices_mod.visualize_C_matrix
visualize_D_matrix = matrices_mod.visualize_D_matrix
visualize_qs = matrices_mod.visualize_qs
visualize_efe_comparison = efe_mod.visualize_efe_comparison
visualize_efe_components = efe_mod.visualize_efe_components
visualize_efe_over_time = efe_mod.visualize_efe_over_time
display_history = history_mod.display_history
display_statistics = stats_mod.display_statistics


# Page configuration
st.set_page_config(
    page_title="IPD Agent Interaction",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title
st.title("🎮 Iterated Prisoner's Dilemma Agent Interaction")
st.markdown("---")

# Initialize session state
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = None
if 'game_history' not in st.session_state:
    st.session_state.game_history = []
if 'agent_wrapper' not in st.session_state:
    st.session_state.agent_wrapper = None
if 'state_history' not in st.session_state:
    st.session_state.state_history = []
if 'step_simulator' not in st.session_state:
    st.session_state.step_simulator = None
if 'simulation_mode' not in st.session_state:
    st.session_state.simulation_mode = 'step'  # 'step' or 'run'

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    # Configure agent
    agent, agent_params = configure_agent()
    
    # Configure opponent
    opponent = configure_opponent()
    
    # Configure game
    game_params = configure_game()
    
    st.markdown("---")
    
    # Simulation mode selection
    simulation_mode = st.radio(
        "Simulation Mode",
        ["Step-by-Step", "Run All"],
        index=0 if st.session_state.simulation_mode == 'step' else 1,
        help="Step-by-Step: Click 'Step' to advance one turn at a time. Run All: Execute entire simulation at once."
    )
    st.session_state.simulation_mode = 'step' if simulation_mode == "Step-by-Step" else 'run'
    
    st.markdown("---")
    
    if st.session_state.simulation_mode == 'step':
        # Step-by-step mode
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔧 Initialize", use_container_width=True):
                # Use the configured agents directly (they're already created with params)
                # Reset them to start fresh
                step_agent = agent
                step_opponent = opponent
                step_agent.reset()
                if hasattr(step_opponent, 'reset'):
                    step_opponent.reset()
                
                # Create wrapper if agent is AIF
                step_wrapper = None
                if hasattr(step_agent, 'agent'):
                    step_wrapper = AIFAgentStateWrapper(step_agent)
                
                st.session_state.step_simulator = StepByStepSimulator(
                    step_agent,
                    step_opponent,
                    game_params['turns'],
                    game_params['noise'],
                    step_wrapper,
                )
                st.session_state.step_simulator.initialize()
                st.session_state.game_history = []
                st.session_state.state_history = []
                st.session_state.agent_wrapper = step_wrapper
                
                # Get initial state
                if step_wrapper:
                    initial_state = step_wrapper.get_current_state(0, (None, None))
                    st.session_state.state_history.append(initial_state)
                
                st.success("Simulation initialized!")
                st.rerun()
        
        with col2:
            if st.button("⏭️ Step", type="primary", use_container_width=True, 
                        disabled=st.session_state.step_simulator is None):
                if st.session_state.step_simulator:
                    state = st.session_state.step_simulator.step()
                    if state:
                        st.session_state.game_history = state['history']
                        if state.get('agent_state'):
                            st.session_state.state_history.append(state['agent_state'])
                        st.session_state.simulation_data = state
                        if state.get('is_complete'):
                            st.success("Simulation completed!")
                        st.rerun()
                    else:
                        st.info("Simulation already complete!")
    
    # Run simulation button (for run-all mode)
    if st.session_state.simulation_mode == 'run' and st.button("🚀 Run Simulation", type="primary", use_container_width=True):
        st.session_state.simulation_running = True
        st.session_state.game_history = []
        st.session_state.state_history = []
        
        # Create wrapper if agent is AIF
        if hasattr(agent, 'agent'):
            st.session_state.agent_wrapper = AIFAgentStateWrapper(agent)
        else:
            st.session_state.agent_wrapper = None
        
        # Reset agents
        agent.reset()
        if hasattr(opponent, 'reset'):
            opponent.reset()
        
        # Run simulation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            for state in simulate_game(
                agent,
                opponent,
                game_params['turns'],
                game_params['noise'],
                game_params['update_interval'],
                st.session_state.agent_wrapper,
            ):
                progress = state['turn'] / state['total_turns']
                progress_bar.progress(progress)
                status_text.text(f"Turn {state['turn']}/{state['total_turns']}")
                
                # Store state
                st.session_state.game_history = state['history']
                if state.get('agent_state'):
                    st.session_state.state_history.append(state['agent_state'])
            
            st.session_state.simulation_data = state
            st.session_state.simulation_running = False
            st.success("Simulation completed!")
            
        except Exception as e:
            st.error(f"Error during simulation: {str(e)}")
            st.session_state.simulation_running = False
    
    # Clear button
    if st.button("🗑️ Clear Results", use_container_width=True):
        st.session_state.simulation_data = None
        st.session_state.game_history = []
        st.session_state.state_history = []
        st.session_state.agent_wrapper = None
        st.session_state.step_simulator = None
        st.rerun()

# Main content area
if st.session_state.simulation_running:
    st.info("Simulation in progress... Please wait.")
elif st.session_state.simulation_data is None and st.session_state.step_simulator is None:
    st.info("👈 Configure agents and game parameters in the sidebar, then click 'Initialize' (for step-by-step) or 'Run Simulation' (for full run) to start.")
elif st.session_state.simulation_data is not None or (st.session_state.step_simulator and st.session_state.step_simulator.current_turn > 0):
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Simulation",
        "📊 History",
        "📈 Evolution",
        "📉 Statistics",
    ])
    
    with tab1:
        st.header("Current Simulation State")
        
        # Get current scores
        current_score = 0.0
        opponent_score = 0.0
        current_turn = 0
        
        if st.session_state.simulation_data:
            current_score = st.session_state.simulation_data.get('my_score', 0.0)
            opponent_score = st.session_state.simulation_data.get('opponent_score', 0.0)
            current_turn = st.session_state.simulation_data.get('turn', 0)
        elif st.session_state.step_simulator:
            current_score = st.session_state.step_simulator.my_score
            opponent_score = st.session_state.step_simulator.opponent_score
            current_turn = st.session_state.step_simulator.current_turn
        
        # Show initial state if simulator is initialized but no steps taken yet
        if (st.session_state.step_simulator and 
            st.session_state.step_simulator.current_turn == 0 and 
            st.session_state.agent_wrapper and 
            not st.session_state.state_history):
            # Get initial state
            initial_state = st.session_state.agent_wrapper.get_current_state(0, (None, None))
            st.session_state.state_history.append(initial_state)
        
        if st.session_state.state_history:
            current_state = st.session_state.state_history[-1]
            
            # Current turn and scores
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Turn", current_turn)
            with col2:
                st.metric("My Score", f"{current_score:.2f}")
            with col3:
                st.metric("Opponent Score", f"{opponent_score:.2f}")
            
            # Step-by-step controls in main area
            if st.session_state.simulation_mode == 'step' and st.session_state.step_simulator:
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    if st.button("⏭️ Step Forward", type="primary", use_container_width=True,
                                disabled=st.session_state.step_simulator.current_turn >= st.session_state.step_simulator.turns):
                        state = st.session_state.step_simulator.step()
                        if state:
                            st.session_state.game_history = state['history']
                            if state.get('agent_state'):
                                st.session_state.state_history.append(state['agent_state'])
                            st.session_state.simulation_data = state
                            st.rerun()
                        else:
                            st.info("Simulation complete!")
                with col2:
                    if st.button("🔄 Reset", use_container_width=True):
                        st.session_state.step_simulator.reset()
                        st.session_state.game_history = []
                        st.session_state.state_history = []
                        st.session_state.simulation_data = None
                        if st.session_state.agent_wrapper:
                            st.session_state.agent_wrapper.clear_cache()
                        st.rerun()
                with col3:
                    if st.session_state.step_simulator:
                        progress = current_turn / st.session_state.step_simulator.turns if st.session_state.step_simulator.turns > 0 else 0
                        st.progress(progress)
                        st.caption(f"Progress: {current_turn}/{st.session_state.step_simulator.turns} turns")
            
            st.markdown("---")
            
            # Matrices
            st.subheader("Agent Matrices")
            
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(visualize_A_matrix(current_state))
            with col2:
                st.pyplot(visualize_C_matrix(current_state))
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(visualize_D_matrix(current_state))
            with col2:
                st.pyplot(visualize_qs(current_state))
            
            st.markdown("---")
            
            # B Matrix
            st.subheader("B Matrix (Transition Model)")
            show_initial = st.checkbox("Show Initial B Matrix", value=False)
            show_diff = st.checkbox("Show Difference", value=False) if show_initial else False
            st.pyplot(visualize_B_matrix(current_state, show_initial, show_diff))
            
            st.markdown("---")
            
            # EFE
            st.subheader("Expected Free Energy (EFE)")
            st.pyplot(visualize_efe_comparison(current_state))
            st.pyplot(visualize_efe_components(current_state))
            
            # Latest moves
            st.markdown("---")
            st.subheader("Latest Moves")
            if st.session_state.game_history:
                latest = st.session_state.game_history[-10:]
                for move in reversed(latest):
                    outcome = move['outcome']
                    st.write(f"Turn {move['turn']}: {outcome} (My: {move['my_payoff']:.1f}, Opponent: {move['opponent_payoff']:.1f})")
        else:
            st.info("No agent state available. This might be a non-AIF agent.")
    
    with tab2:
        st.header("Game History")
        display_history(st.session_state.game_history)
    
    with tab3:
        st.header("Matrix Evolution")
        
        if st.session_state.state_history:
            # B Matrix evolution
            st.subheader("B Matrix Evolution")
            st.info("B matrix evolution visualization coming soon!")
            
            # EFE over time
            st.subheader("EFE Evolution")
            efe_fig = visualize_efe_over_time(st.session_state.state_history)
            if efe_fig:
                st.pyplot(efe_fig)
        else:
            st.info("No state history available. This might be a non-AIF agent.")
    
    with tab4:
        st.header("Statistics Dashboard")
        display_statistics(st.session_state.game_history)

# Footer
st.markdown("---")
st.markdown("**IPD Agent Interaction** - Explore Active Inference and other learning agents in the Iterated Prisoner's Dilemma")

