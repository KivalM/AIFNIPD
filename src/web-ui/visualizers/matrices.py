"""Matrix visualization components."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from typing import Dict, Any

# Import constants - handle both absolute and relative imports
try:
    from agents.aif.jax.five_state import (
        ALL_STATES_LABELS,
        ALL_OBS_LABELS,
        ALL_ACTIONS_LABELS,
    )
except ImportError:
    # Fallback if running as module
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(src_path))
    from agents.aif.jax.five_state import (
        ALL_STATES_LABELS,
        ALL_OBS_LABELS,
        ALL_ACTIONS_LABELS,
    )


def plot_matrix_heatmap(matrix: np.ndarray, title: str, x_labels: list = None, y_labels: list = None):
    """Create a heatmap plot for a matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='viridis', cbar=True, ax=ax)
    
    if x_labels:
        ax.set_xticks([i + 0.5 for i in range(len(x_labels))])
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
    if y_labels:
        ax.set_yticks([i + 0.5 for i in range(len(y_labels))])
        ax.set_yticklabels(y_labels)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_A_matrix(state: Dict[str, Any]):
    """Visualize A matrix (observation model)."""
    A = state['A']
    fig = plot_matrix_heatmap(
        A,
        'A Matrix: Observation Model',
        x_labels=ALL_STATES_LABELS,
        y_labels=ALL_OBS_LABELS,
    )
    return fig


def visualize_B_matrix(state: Dict[str, Any], show_initial: bool = False, show_diff: bool = False):
    """Visualize B matrix (transition model) for both actions."""
    B = state['B']
    B_initial = state.get('B_initial', B)
    
    if show_initial and show_diff:
        # Show three plots: current, initial, difference
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for action_idx, action_label in enumerate(ALL_ACTIONS_LABELS):
            # Current B
            sns.heatmap(
                B[:, :, action_idx],
                annot=True,
                fmt='.3f',
                cmap='viridis',
                cbar=True,
                ax=axes[0] if action_idx == 0 else axes[1],
                xticklabels=ALL_STATES_LABELS,
                yticklabels=ALL_STATES_LABELS,
            )
            if action_idx == 0:
                axes[0].set_title(f'B Matrix (Current) - Action {action_label}', fontweight='bold')
            else:
                axes[1].set_title(f'B Matrix (Current) - Action {action_label}', fontweight='bold')
            
            # Initial B
            sns.heatmap(
                B_initial[:, :, action_idx],
                annot=True,
                fmt='.3f',
                cmap='viridis',
                cbar=True,
                ax=axes[2] if action_idx == 0 else None,
                xticklabels=ALL_STATES_LABELS,
                yticklabels=ALL_STATES_LABELS,
            )
            if action_idx == 0:
                axes[2].set_title('B Matrix (Initial) - Action C', fontweight='bold')
        
        # Difference
        diff = B - B_initial
        sns.heatmap(
            diff[:, :, 0],
            annot=True,
            fmt='.3f',
            cmap='RdBu_r',
            center=0,
            cbar=True,
            ax=axes[2],
            xticklabels=ALL_STATES_LABELS,
            yticklabels=ALL_STATES_LABELS,
        )
        axes[2].set_title('B Matrix Difference (Current - Initial)', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    # Show current B for both actions side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for action_idx, action_label in enumerate(ALL_ACTIONS_LABELS):
        sns.heatmap(
            B[:, :, action_idx],
            annot=True,
            fmt='.3f',
            cmap='viridis',
            cbar=True,
            ax=axes[action_idx],
            xticklabels=ALL_STATES_LABELS,
            yticklabels=ALL_STATES_LABELS,
        )
        axes[action_idx].set_title(f'B Matrix - Action {action_label}', fontweight='bold')
        axes[action_idx].set_xlabel('Next State')
        axes[action_idx].set_ylabel('Current State')
    
    plt.tight_layout()
    return fig


def visualize_C_matrix(state: Dict[str, Any]):
    """Visualize C matrix (preferences) as bar chart."""
    C = state['C']
    # Ensure C is a 1D numpy array
    C = np.array(C)
    if C.ndim > 1:
        C = C.flatten()
    # Convert to Python float list to avoid JAX array issues
    C = [float(x) for x in C]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(C)), C, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(C)))
    ax.set_xticklabels(ALL_OBS_LABELS)
    ax.set_ylabel('Preference Value', fontsize=11)
    ax.set_title('C Matrix: Preferences', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, C)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig


def visualize_D_matrix(state: Dict[str, Any]):
    """Visualize D matrix (initial state distribution) as bar chart."""
    D = state['D']
    # Ensure D is a 1D numpy array
    D = np.array(D)
    if D.ndim > 1:
        D = D.flatten()
    # Convert to Python float list to avoid JAX array issues
    D = [float(x) for x in D]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(D)), D, color='coral', alpha=0.7)
    ax.set_xticks(range(len(D)))
    ax.set_xticklabels(ALL_STATES_LABELS)
    ax.set_ylabel('Probability', fontsize=11)
    ax.set_title('D Matrix: Initial State Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, D)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig


def visualize_qs(state: Dict[str, Any]):
    """Visualize current state beliefs (qs) as bar chart."""
    qs = state['qs']
    # Ensure qs is a 1D numpy array
    qs = np.array(qs)
    if qs.ndim > 1:
        qs = qs.flatten()
    # Convert to Python float list to avoid JAX array issues
    qs = [float(x) for x in qs]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(qs)), qs, color='mediumpurple', alpha=0.7)
    ax.set_xticks(range(len(qs)))
    ax.set_xticklabels(ALL_STATES_LABELS)
    ax.set_ylabel('Belief Probability', fontsize=11)
    ax.set_title('Current State Beliefs (qs)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, qs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig

