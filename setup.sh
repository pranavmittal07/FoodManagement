#!/bin/bash
# Install Python and dependencies
pip install -r requirements.txt
# Install Streamlit (if not included in requirements.txt)
pip install streamlit
# Run the app
streamlit run app.py
