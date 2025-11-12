"""Statistics dashboard component."""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any


def calculate_statistics(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate aggregate statistics from game history."""
    if not history:
        return {}
    
    df = pd.DataFrame(history)
    
    # Count outcomes
    outcomes = df['outcome'].value_counts()
    total_turns = len(df)
    
    cc_count = outcomes.get('CC', 0)
    cd_count = outcomes.get('CD', 0)
    dc_count = outcomes.get('DC', 0)
    dd_count = outcomes.get('DD', 0)
    
    # Calculate rates
    cc_rate = cc_count / total_turns if total_turns > 0 else 0
    cd_rate = cd_count / total_turns if total_turns > 0 else 0
    dc_rate = dc_count / total_turns if total_turns > 0 else 0
    dd_rate = dd_count / total_turns if total_turns > 0 else 0
    
    # Normalized cooperation (CC / (CC + CD))
    total_coop = cc_count + cd_count
    norm_coop = cc_count / total_coop if total_coop > 0 else 0
    
    # Final scores
    my_final_score = df['my_score'].iloc[-1] if len(df) > 0 else 0
    opponent_final_score = df['opponent_score'].iloc[-1] if len(df) > 0 else 0
    
    # Average payoffs
    my_avg_payoff = df['my_payoff'].mean()
    opponent_avg_payoff = df['opponent_payoff'].mean()
    
    return {
        'total_turns': total_turns,
        'cc_count': cc_count,
        'cd_count': cd_count,
        'dc_count': dc_count,
        'dd_count': dd_count,
        'cc_rate': cc_rate,
        'cd_rate': cd_rate,
        'dc_rate': dc_rate,
        'dd_rate': dd_rate,
        'norm_coop': norm_coop,
        'my_final_score': my_final_score,
        'opponent_final_score': opponent_final_score,
        'my_avg_payoff': my_avg_payoff,
        'opponent_avg_payoff': opponent_avg_payoff,
    }


def display_statistics(history: List[Dict[str, Any]]):
    """Display statistics dashboard."""
    if not history:
        st.info("No game history yet. Run a simulation to see statistics.")
        return
    
    stats = calculate_statistics(history)
    
    # Key metrics
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("My Final Score", f"{stats['my_final_score']:.2f}")
        st.metric("My Avg Payoff", f"{stats['my_avg_payoff']:.3f}")
    with col2:
        st.metric("Opponent Final Score", f"{stats['opponent_final_score']:.2f}")
        st.metric("Opponent Avg Payoff", f"{stats['opponent_avg_payoff']:.3f}")
    with col3:
        st.metric("CC Rate", f"{stats['cc_rate']:.2%}")
        st.metric("Normalized Cooperation", f"{stats['norm_coop']:.2%}")
    with col4:
        st.metric("Total Turns", stats['total_turns'])
        st.metric("CD Rate", f"{stats['cd_rate']:.2%}")
    
    # Outcome distribution
    st.subheader("Outcome Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = ['CC', 'CD', 'DC', 'DD']
        sizes = [stats['cc_count'], stats['cd_count'], stats['dc_count'], stats['dd_count']]
        colors = ['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6']
        explode = (0.05, 0.05, 0.05, 0.05)
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90)
        ax.set_title('Outcome Distribution', fontsize=12, fontweight='bold')
        st.pyplot(fig)
    
    with col2:
        # Bar chart
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(labels, sizes, color=colors, alpha=0.7)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Outcome Counts', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, sizes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(val)}',
                   ha='center', va='bottom', fontsize=10)
        st.pyplot(fig)
    
    # Score evolution
    st.subheader("Score Evolution")
    df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['turn'], df['my_score'], label='My Score', linewidth=2)
    ax.plot(df['turn'], df['opponent_score'], label='Opponent Score', linewidth=2)
    ax.set_xlabel('Turn', fontsize=11)
    ax.set_ylabel('Cumulative Score', fontsize=11)
    ax.set_title('Score Evolution Over Time', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Detailed statistics table
    st.subheader("Detailed Statistics")
    stats_df = pd.DataFrame([
        {'Metric': 'CC Count', 'Value': stats['cc_count']},
        {'Metric': 'CD Count', 'Value': stats['cd_count']},
        {'Metric': 'DC Count', 'Value': stats['dc_count']},
        {'Metric': 'DD Count', 'Value': stats['dd_count']},
        {'Metric': 'CC Rate', 'Value': f"{stats['cc_rate']:.2%}"},
        {'Metric': 'CD Rate', 'Value': f"{stats['cd_rate']:.2%}"},
        {'Metric': 'DC Rate', 'Value': f"{stats['dc_rate']:.2%}"},
        {'Metric': 'DD Rate', 'Value': f"{stats['dd_rate']:.2%}"},
        {'Metric': 'Normalized Cooperation', 'Value': f"{stats['norm_coop']:.2%}"},
    ])
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

