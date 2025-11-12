# IPD Agent Web UI

A Streamlit-based web interface for interacting with Iterated Prisoner's Dilemma agents, with a focus on Active Inference (AIF) agents.

## Features

- **Interactive Agent Configuration**: Configure AIF agents and other learning agents with customizable parameters
- **Real-time Visualization**: View A, B, C, D matrices and Expected Free Energy (EFE) as the game progresses
- **Game History**: Track turn-by-turn game history with detailed statistics
- **Matrix Evolution**: Observe how the B matrix changes over time as the agent learns
- **Statistics Dashboard**: Comprehensive statistics including cooperation rates, scores, and outcome distributions

## Running the App

From the project root directory:

```bash
streamlit run src/web-ui/app.py
```

Or using the run script:

```bash
python src/web-ui/run.py
```

## Usage

1. **Configure Agent**: Select agent type and adjust parameters in the sidebar
2. **Configure Opponent**: Choose an opponent (another agent or classic strategy)
3. **Set Game Parameters**: Set number of turns, noise level, and visualization update interval
4. **Run Simulation**: Click "Run Simulation" to start
5. **Explore Results**: Navigate through tabs to view:
   - **Simulation**: Current matrices, EFE, and latest moves
   - **History**: Turn-by-turn game history
   - **Evolution**: Matrix evolution over time
   - **Statistics**: Aggregate statistics and charts

## Supported Agents

### Active Inference Agents
- AIF (Five State)
- AIF (Five State Noisy)
- AIF (Five State Utility)

### Learning Agents
- Q-Learning
- Cooperative Q-Learning
- Bayesian Q-Learning
- Cooperative Bayesian Q-Learning
- PSRL (Posterior Sampling for Reinforcement Learning)
- Cooperative PSRL
- DynaQ
- Cooperative DynaQ

### Classic Strategies
- TitForTat
- Always Cooperate
- Always Defect
- Random
- Grudger
- DBS

## Architecture

The UI uses a wrapper pattern (`AIFAgentStateWrapper`) to extract and cache agent state efficiently. The wrapper provides a `get_current_state()` method that returns:
- A, B, C, D matrices
- Current state beliefs (qs)
- Expected Free Energy for both actions
- EFE component breakdowns

This allows for efficient real-time visualization without redundant computations.

