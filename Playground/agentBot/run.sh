#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
else
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi

# Run the shell agent
python shell_agent.py
