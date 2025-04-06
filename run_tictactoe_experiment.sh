#!/bin/bash

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add the project root to PYTHONPATH
export PYTHONPATH=$DIR:$PYTHONPATH

echo "Running TicTacToe Memory Experiment..."
echo "Make sure you have set the OPENAI_API_KEY environment variable or updated the .env file!"
echo "You can specify board size with --board_size [3,4,5,6]"

# Run the experiment with provided args
python3 $DIR/experiments/run_self_play.py "$@" 