"""Wrapper class for extracting state from AIF agents."""

import jax.numpy as jnp
import numpy as np
from typing import Dict, Optional, Tuple, Any
from axelrod.action import Action

# Import constants - handle both absolute and relative imports
try:
    from agents.aif.jax.five_state import (
        START, CC, CD, DC, DD,
        C, D,
        ALL_STATES_LABELS,
        ALL_OBS_LABELS,
        ALL_ACTIONS_LABELS,
        action_pair_to_obs,
    )
except ImportError:
    # Fallback if running as module
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(src_path))
    from agents.aif.jax.five_state import (
        START, CC, CD, DC, DD,
        C, D,
        ALL_STATES_LABELS,
        ALL_OBS_LABELS,
        ALL_ACTIONS_LABELS,
        action_pair_to_obs,
    )


class AIFAgentStateWrapper:
    """Wrapper for AIF agents that provides state extraction and caching."""
    
    def __init__(self, aif_agent):
        """
        Initialize wrapper for an AIF agent.
        
        Args:
            aif_agent: Instance of JaxFiveStateAgent or similar AIF agent
        """
        self.agent = aif_agent
        self._cache: Dict[int, Dict[str, Any]] = {}
        self._last_turn = -1
        self._initial_B = None
        self._state_history: list = []
        
    def get_current_state(self, turn_number: int, current_observation: Optional[Tuple[Optional[Action], Optional[Action]]] = None) -> Dict[str, Any]:
        """
        Get current state of the agent with caching.
        
        Args:
            turn_number: Current turn number
            current_observation: Current joint state (my_action, opponent_action) or None for initial state
            
        Returns:
            Dictionary containing:
            - A: Observation model matrix
            - B: Transition model matrices (for both actions)
            - C: Preference vector
            - D: Initial state distribution
            - qs: Current state beliefs
            - efe: Expected Free Energy for both actions
            - efe_components: Breakdown of EFE components
        """
        # Return cached if already computed for this turn
        if turn_number in self._cache:
            return self._cache[turn_number]
        
        # Extract matrices from the internal AIF agent
        aif_agent = self.agent.agent
        
        # Get A matrix (observation model) - squeeze out batch dimension if present
        A = np.array(aif_agent.A[0])  # Shape: (num_obs, num_states) or (1, num_obs, num_states)
        if A.ndim == 3:
            A = A.squeeze(0)  # Remove batch dimension
        
        # Get B matrix (transition model) - shape: (num_states, num_states, num_actions)
        B = np.array(aif_agent.B[0])
        if B.ndim == 4:
            B = B.squeeze(0)  # Remove batch dimension if present
        
        # Store initial B if this is the first call
        if self._initial_B is None:
            self._initial_B = B.copy()
        
        # Get C matrix (preferences) - squeeze out batch dimension if present
        C = np.array(aif_agent.C[0])  # Shape: (num_obs,) or (1, num_obs)
        if C.ndim == 2:
            C = C.squeeze(0)  # Remove batch dimension
        
        # Get D matrix (initial state distribution) - squeeze out batch dimension if present
        D = np.array(aif_agent.D[0])  # Shape: (num_states,) or (1, num_states)
        if D.ndim == 2:
            D = D.squeeze(0)  # Remove batch dimension
        
        # Get current beliefs (qs) - squeeze out batch dimension if present
        qs = None
        if hasattr(self.agent, 'qs') and len(self.agent.qs) > 0:
            # Get the most recent belief
            qs = np.array(self.agent.qs[-1][0])  # Shape: (num_states,) or (1, num_states)
            if qs.ndim == 2:
                qs = qs.squeeze(0)  # Remove batch dimension
        else:
            # Use empirical prior or D
            if hasattr(self.agent, 'empirical_prior'):
                qs = np.array(self.agent.empirical_prior[0])
                if qs.ndim == 2:
                    qs = qs.squeeze(0)  # Remove batch dimension
            else:
                qs = D.copy()
        
        # Always compute EFE directly from infer_policies and aggregate by first action
        efe_C = None
        efe_D = None
        efe_components_C = {}
        efe_components_D = {}

        # Use the raw belief object for infer_policies if available; else build from current qs
        try:
            if hasattr(self.agent, 'qs') and len(self.agent.qs) > 0:
                qs_for_infer = self.agent.qs[-1]  # Use raw structure saved by the agent
            elif hasattr(self.agent, 'empirical_prior'):
                qs_for_infer = self.agent.empirical_prior
            else:
                # Fallback: construct a broadcasted prior from D
                qs_for_infer = [jnp.broadcast_to(jnp.array(D)[None, :], (1, D.shape[0]))]
            
            qpi_raw, efe_raw = aif_agent.infer_policies(qs_for_infer)
            # Convert to 1D numpy arrays
            qpi_arr = np.array(qpi_raw)
            efe_arr = np.array(efe_raw)
            qpi_arr = np.squeeze(qpi_arr)
            efe_arr = np.squeeze(efe_arr)
            if qpi_arr.ndim > 1:
                qpi_arr = qpi_arr.reshape(-1)
            if efe_arr.ndim > 1:
                efe_arr = efe_arr.reshape(-1)
            num_policies = int(efe_arr.shape[0])
            # Map policies to first action: assume lexicographic enumeration -> first half: C, second half: D
            half = max(1, num_policies // 2)
            # Guard against degenerate cases
            qpi_arr = qpi_arr[:num_policies]
            efe_arr = efe_arr[:num_policies]
            # Weighted average EFE per first action
            wC = qpi_arr[:half]
            wD = qpi_arr[half:num_policies]
            gC = efe_arr[:half]
            gD = efe_arr[half:num_policies]
            # Normalize weights to avoid zero-division
            wC_sum = float(wC.sum()) if wC.size else 0.0
            wD_sum = float(wD.sum()) if wD.size else 0.0
            if wC_sum > 0:
                efe_C = float((wC * gC).sum() / wC_sum)
            elif gC.size:
                efe_C = float(gC.mean())
            else:
                efe_C = 0.0
            if wD_sum > 0:
                efe_D = float((wD * gD).sum() / wD_sum)
            elif gD.size:
                efe_D = float(gD.mean())
            else:
                efe_D = 0.0
            # Components: we only expose total for now to avoid inconsistent decomposition
            efe_components_C = {'total': efe_C}
            efe_components_D = {'total': efe_D}
        except Exception:
            # Fallback to simplified manual computation if infer_policies path fails
            efe_C, efe_components_C = self._compute_efe(aif_agent, qs, action_idx=0)
            efe_D, efe_components_D = self._compute_efe(aif_agent, qs, action_idx=1)
        
        # Build state dictionary
        state = {
            'turn': turn_number,
            'A': A,
            'B': B,
            'B_initial': self._initial_B,
            'C': C,
            'D': D,
            'qs': qs,
            'efe': {
                'C': efe_C,
                'D': efe_D,
            },
            'efe_components': {
                'C': efe_components_C,
                'D': efe_components_D,
            },
        }
        
        # Cache the state
        self._cache[turn_number] = state
        self._state_history.append(state)
        self._last_turn = turn_number
        
        return state
    
    def _compute_efe(self, aif_agent, qs: np.ndarray, action_idx: int) -> Tuple[float, Dict[str, float]]:
        """
        Compute Expected Free Energy for a given action.
        
        This is a simplified computation. The actual EFE in pymdp includes:
        - Expected utility (from C matrix)
        - State information gain
        - Parameter information gain
        
        Args:
            aif_agent: The internal AIF agent
            qs: Current state beliefs
            action_idx: Action index (0 for C, 1 for D)
            
        Returns:
            Tuple of (total_efe, components_dict)
        """
        # Get B matrix for this action - handle batch dimension
        B_full = np.array(aif_agent.B[0])
        if B_full.ndim == 4:
            B_full = B_full.squeeze(0)  # Remove batch dimension
        B_action = B_full[:, :, action_idx]  # Shape: (num_states, num_states)
        
        # Get C matrix (preferences) - handle batch dimension
        C_prefs = np.array(aif_agent.C[0])
        if C_prefs.ndim == 2:
            C_prefs = C_prefs.squeeze(0)  # Remove batch dimension
        
        # Ensure qs is 1D
        if qs.ndim > 1:
            qs = qs.squeeze()
        
        # Expected next state distribution given current beliefs and action
        # p(s'|s, a) * p(s)
        expected_next_state = B_action.T @ qs  # Shape: (num_states,)
        
        # Expected observation distribution (A maps states to observations) - handle batch dimension
        A_matrix = np.array(aif_agent.A[0])
        if A_matrix.ndim == 3:
            A_matrix = A_matrix.squeeze(0)  # Remove batch dimension
        expected_obs = A_matrix @ expected_next_state  # Shape: (num_obs,)
        
        # Expected utility: sum over observations of p(o|a) * C(o)
        expected_utility = np.sum(expected_obs * C_prefs)
        
        # State information gain: KL divergence between expected next state and prior
        # Using uniform prior as reference (or could use D)
        prior = np.ones_like(expected_next_state) / len(expected_next_state)
        kl_state = np.sum(expected_next_state * np.log(expected_next_state / (prior + 1e-10) + 1e-10))
        
        # Parameter information gain (simplified - would need access to pB)
        # For now, use a placeholder based on uncertainty in B
        param_info_gain = 0.0  # Would need pB to compute properly
        
        # Total EFE (negative because we want to minimize it, but display as positive)
        # In Active Inference, we minimize EFE, so lower is better
        total_efe = -expected_utility + kl_state + param_info_gain
        
        components = {
            'expected_utility': float(expected_utility),
            'state_info_gain': float(kl_state),
            'param_info_gain': float(param_info_gain),
            'total': float(total_efe),
        }
        
        return float(total_efe), components
    
    def get_state_history(self) -> list:
        """Get history of all cached states."""
        return self._state_history.copy()
    
    def clear_cache(self):
        """Clear the cache (useful when resetting agent)."""
        self._cache.clear()
        self._state_history.clear()
        self._last_turn = -1
        self._initial_B = None

