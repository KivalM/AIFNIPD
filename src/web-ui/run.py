#!/usr/bin/env python3
"""Entry point for running the Streamlit web UI."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

import streamlit.web.cli as stcli

if __name__ == "__main__":
    app_path = src_path / "web-ui" / "app.py"
    sys.argv = ["streamlit", "run", str(app_path)]
    sys.exit(stcli.main())

