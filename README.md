# Research on LLM Memory and Learning through interaction in Multi-Agent Systems

## Research Focus
This research project investigates how large language models (LLMs) build knowledge through multi-agent interactions. The focus is on testing how LLMs use memory, inference, and retrieval mechanisms to enhance their performance across iterative games of TicTacToe.

## Project Structure
```
.
├── docs                    # Documentation and research notes
├── experiments             # Experimental scripts and configurations
│   ├── agent.py           # TicTacToe agent implementation
│   ├── memory.py          # Memory manager for storing and retrieving information
│   ├── run_self_play.py   # Script for running self-play experiments
│   ├── tictactoe.py       # TicTacToe game implementation
│   └── utils.py           # Utility functions for experiments
└── analysis               # Analysis scripts and notebooks for experimental results
    └── visualize.py       # Script for visualizing experimental results
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- An OpenAI API key

### Installation
1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment: 
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Set up your OpenAI API key in a `.env` file in the root directory:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Running Experiments
To run a self-play experiment:
```bash
python experiments/run_self_play.py --num_games 10 --agent_a_model gpt-4-0125-preview --agent_b_model gpt-4-0125-preview
```

## Latest Progress

### Experimental Framework Setup
- Implemented TicTacToe game environment with variable board sizes (3x3 to 6x6)
- Developed agent framework with memory management capabilities
- Created logging system for tracking game statistics and agent performance
- Set up self-play experiment runner with configurable parameters

### Memory Function Implementation
- Added memory functions for storing and retrieving game information
- Implemented strategies for pattern recognition and move planning
- Created functions for analyzing opponent behavior and adapting strategies

### Data Collection and Analysis
- Built logging system to track agent decisions and reasoning
- Developed metrics for evaluating memory usage efficiency
- Created visualizations for analyzing agent performance over multiple games

### Enhanced Memory Mechanisms
- Implemented hierarchical memory structure
- Added support for explicit pattern identification
- Created mechanisms for strategic knowledge transfer between games
- Established methods for storing and recognizing opponent patterns

### Experiments with Different Board Sizes
- Expanded game environment to support board sizes from 3x3 to 6x6
- Added configuration for testing memory transfer across different board complexities
- Implemented adaptive strategies based on board size

### Integration of Different Interaction Methods
- Added support for both function calling and board submission methods
- Implemented tracking for token usage and memory calls
- Created comparative analysis framework for different interaction paradigms

## Running the Code

### Basic Self-Play
```bash
python experiments/run_self_play.py --num_games 10 --agent_a_model gpt-4-0125-preview --agent_b_model gpt-4-0125-preview
```

### With Custom Configuration
```bash
python experiments/run_self_play.py --num_games 20 --agent_a_model gpt-4-0125-preview --agent_b_model gpt-3.5-turbo-0125 --board_size 4 --first_player random --output_dir results/experiment_1
```

### Options
- `--num_games`: Number of games to play (default: 10)
- `--agent_a_model`: Model for Agent A (default: gpt-4-0125-preview)
- `--agent_b_model`: Model for Agent B (default: gpt-4-0125-preview)
- `--board_size`: Board size (3-6, default: 3)
- `--first_player`: Who plays first - 'X', 'O', or 'random' (default: X)
- `--output_dir`: Directory to save results (default: experiments/results)
- `--use_board_submit`: Use board submission method (default: False)
- `--train_autoencoder`: Train an autoencoder on board states (default: False)
