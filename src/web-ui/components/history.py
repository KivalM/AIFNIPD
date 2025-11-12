"""Game history display component."""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any

# Handle imports - try relative first, then absolute
try:
    from ..utils import create_history_dataframe, get_outcome_color
except (ImportError, ValueError):
    # Fallback for when loaded via importlib
    import sys
    from pathlib import Path
    # Get the web-ui directory (parent of components)
    web_ui_path = Path(__file__).resolve().parent.parent
    if str(web_ui_path) not in sys.path:
        sys.path.insert(0, str(web_ui_path))
    from utils import create_history_dataframe, get_outcome_color


def display_history(history: List[Dict[str, Any]]):
    """Display game history as a styled dataframe."""
    if not history:
        st.info("No game history yet. Run a simulation to see history.")
        return
    
    df = create_history_dataframe(history)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Turns", len(df))
    with col2:
        st.metric("My Score", f"{df['My Score'].iloc[-1]:.2f}")
    with col3:
        st.metric("Opponent Score", f"{df['Opponent Score'].iloc[-1]:.2f}")
    with col4:
        cc_count = len(df[df['Outcome'] == 'CC'])
        st.metric("CC Rate", f"{cc_count / len(df):.2%}")
    
    # Style the dataframe
    def style_outcome(val):
        color = get_outcome_color(val)
        return f'background-color: {color}; color: white; font-weight: bold'
    
    styled_df = df.style.applymap(style_outcome, subset=['Outcome'])
    
    # Display dataframe
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download History as CSV",
        data=csv,
        file_name="game_history.csv",
        mime="text/csv",
    )

