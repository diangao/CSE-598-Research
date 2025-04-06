#!/bin/bash

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add the project root to PYTHONPATH
export PYTHONPATH=$DIR:$PYTHONPATH

# Activate virtual environment if it exists
if [ -d "$DIR/tictactoe_env" ]; then
    source "$DIR/tictactoe_env/bin/activate"
    echo "Virtual environment activated"
fi

# Install tabulate if not present
pip install tabulate --quiet > /dev/null 2>&1

echo "Showing Experiment Results..."

# Run the statistics display script with provided args
python3 $DIR/experiments/show_latest_results.py "$@" 