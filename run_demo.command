#!/bin/bash
# Double-click this file in Finder to launch the Wane demo in your browser.
cd "$(dirname "$0")"
.venv/bin/streamlit run app.py
