"""EFE (Expected Free Energy) visualization components."""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from typing import Dict, Any, List


def visualize_efe_comparison(state: Dict[str, Any]):
    """Visualize EFE comparison for C vs D actions."""
    efe = state['efe']
    efe_C = efe['C']
    efe_D = efe['D']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(['Cooperate (C)', 'Defect (D)'], [efe_C, efe_D], 
                   color=['#2ecc71', '#e74c3c'], alpha=0.7)
    ax.set_ylabel('Expected Free Energy', fontsize=11)
    ax.set_title('Expected Free Energy Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, [efe_C, efe_D]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add note about lower being better
    ax.text(0.5, 0.95, 'Lower EFE is better (minimization)', 
            transform=ax.transAxes, ha='center', fontsize=9, 
            style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def visualize_efe_components(state: Dict[str, Any]):
    """Visualize EFE components breakdown as stacked bar chart."""
    components_C = state['efe_components']['C']
    components_D = state['efe_components']['D']
    
    # Safely extract components; fall back to 0.0 if not provided
    c_eu = float(components_C.get('expected_utility', 0.0)) if isinstance(components_C, dict) else 0.0
    c_sig = float(components_C.get('state_info_gain', 0.0)) if isinstance(components_C, dict) else 0.0
    c_pig = float(components_C.get('param_info_gain', 0.0)) if isinstance(components_C, dict) else 0.0
    d_eu = float(components_D.get('expected_utility', 0.0)) if isinstance(components_D, dict) else 0.0
    d_sig = float(components_D.get('state_info_gain', 0.0)) if isinstance(components_D, dict) else 0.0
    d_pig = float(components_D.get('param_info_gain', 0.0)) if isinstance(components_D, dict) else 0.0
    missing_components = (
        ('expected_utility' not in components_C or 'state_info_gain' not in components_C or 'param_info_gain' not in components_C) or
        ('expected_utility' not in components_D or 'state_info_gain' not in components_D or 'param_info_gain' not in components_D)
    ) if isinstance(components_C, dict) and isinstance(components_D, dict) else True
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Action C components
    labels = ['Expected\nUtility', 'State Info\nGain', 'Param Info\nGain']
    values_C = [c_eu, c_sig, c_pig]
    colors_C = ['#3498db', '#9b59b6', '#e67e22']
    
    bars1 = ax1.bar(labels, values_C, color=colors_C, alpha=0.7)
    ax1.set_ylabel('Value', fontsize=11)
    ax1.set_title('EFE Components - Action C', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='black', linewidth=0.8)
    
    # Add value labels
    for bar, val in zip(bars1, values_C):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    # Action D components
    values_D = [d_eu, d_sig, d_pig]
    
    bars2 = ax2.bar(labels, values_D, color=colors_C, alpha=0.7)
    ax2.set_ylabel('Value', fontsize=11)
    ax2.set_title('EFE Components - Action D', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linewidth=0.8)
    
    # Add value labels
    for bar, val in zip(bars2, values_D):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    # If components are missing, add a note
    if missing_components:
        fig.suptitle('Note: Detailed EFE components not available for this step; zeros shown.', fontsize=10, color='gray')
    
    plt.tight_layout()
    return fig


def visualize_efe_over_time(state_history: List[Dict[str, Any]]):
    """Visualize EFE evolution over time."""
    if not state_history:
        return None
    
    turns = [s['turn'] for s in state_history]
    efe_C = [s['efe']['C'] for s in state_history]
    efe_D = [s['efe']['D'] for s in state_history]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(turns, efe_C, marker='o', label='EFE (C)', linewidth=2, markersize=4)
    ax.plot(turns, efe_D, marker='s', label='EFE (D)', linewidth=2, markersize=4)
    ax.set_xlabel('Turn', fontsize=11)
    ax.set_ylabel('Expected Free Energy', fontsize=11)
    ax.set_title('EFE Evolution Over Time', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

