# TicTacToe Self-Play Memory Experiment

This experiment investigates how two GPT-3.5 agents with different optimization objectives learn to use external memory strategically when playing TicTacToe against each other.

## Experiment Design

**Agent A**: Optimized solely to maximize win rate
**Agent B**: Optimized for a tradeoff between winning and token efficiency

Both agents have access to three memory types:
- **GraphMemory**: For structured relationships and transitions
- **VectorMemory**: For pattern matching and similarity search
- **SemanticMemory**: For conceptual knowledge and strategic reasoning

The experiment tracks how agents learn to use memory strategically through multiple games, including which types of memory they prefer, when they choose to store vs. retrieve information, and how they structure their memories.

## Task Complexity Progression (Board Size Scaling)

To test how different memory architectures support generalization and adaptability under increasing task complexity, we implement a complexity gradient by progressively enlarging the TicTacToe board:

- **3×3**: Baseline (short horizon, low memory pressure)
- **6×6**: Optional stretch goal (scaling limit test)
- **9x9**: Advanced complexity level

Each size introduces exponentially more possible states and longer action sequences, creating increasing pressure on the agent's memory retrieval and schema organization capabilities.

## Experimental Goals at Each Complexity Level

At each board size, we measure:

- Win rate
- Token efficiency
- Memory usage frequency and diversity (which memory modules are accessed)
- Retrieval usefulness (whether retrieved memory affected decisions)
- Schema update frequency (did the agent adapt memory structure?)
- Retrieval reuse (does prior memory transfer between tasks?)

This setup allows us to test both robustness (can agents reuse the same schema across tasks?) and adaptation (do agents restructure memory to suit the harder task?).

## Setup

1. Install the required dependencies:
   ```
   pip install -r experiments/requirements.txt
   ```

2. Set your OpenAI API key:
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```

## Running the Experiment

```bash
python experiments/run_self_play.py --num_games 10
```

### Command-Line Options

- `--num_games N`: Number of games to run (default: 10)
- `--openai_api_key KEY`: Your OpenAI API key (alternatively, set the OPENAI_API_KEY environment variable)
- `--first_player [agent_a|agent_b|random]`: Which agent goes first (default: random)
- `--output_dir DIR`: Directory to store results (default: experiments/results)
- `--memory_reset [none|game|session]`: When to reset agent memories (default: none)
  - `none`: Memories persist across all games
  - `game`: Memories reset after each game
  - `session`: Memories only reset at the start of the experiment
- `--board_size N`: Size of the TicTacToe board (N×N) (default: 3, options: 3, 4, 5, 6)

## Example Commands

Run 5 games with agent_a always going first:
```bash
python experiments/run_self_play.py --num_games 5 --first_player agent_a
```

Run 20 games with random first player and memory reset between games:
```bash
python experiments/run_self_play.py --num_games 20 --memory_reset game
```

Run 10 games with a 4×4 board:
```bash
python experiments/run_self_play.py --num_games 10 --board_size 4
```

## Output

The experiment saves detailed logs for each game in JSON format under `experiments/results/game_logs/`. These logs include:
- Per-turn actions and memory usage
- Memory function calls and content
- Schema updates
- Token usage
- Game outcomes
- Board size

A summary statistics file is also generated at the end of the experiment run. 

## Memory Comparison Baseline Experiments

To compare the performance of agents with and without memory, use the `run_memory_baselines.py` script:

```bash
python experiments/run_memory_baselines.py --games 15
```

This script automatically runs several comparison experiments:
1. **Agent A with memory vs Agent A without memory** - Tests the impact of memory on Agent A
2. **Agent B with memory vs Agent B without memory** - Tests the impact of memory on Agent B
3. **Agent A without memory vs Agent B without memory** - Baseline comparison with no memory

### Command-Line Options for Memory Baselines

- `--games N`: Number of games per condition (default: 15)
- `--results-dir DIR`: Directory to save results (default: experiments/results/memory_baseline)
- `--model MODEL`: LLM model to use (default: gpt-4o-mini)
- `--skip-3x3`: Skip 3x3 board experiments
- `--skip-9x9`: Skip 9x9 board experiments
- `--skip-graph`: Skip GraphMemory experiments
- `--skip-vector`: Skip VectorMemory experiments
- `--skip-no-memory`: Skip No-Memory baseline experiments

### Example Commands for Memory Baselines

Run only 3x3 board experiments with 10 games each:
```bash
python experiments/run_memory_baselines.py --games 10 --skip-9x9
```

Run only vector memory comparison experiments with GPT-4:
```bash
python experiments/run_memory_baselines.py --skip-graph --skip-no-memory --model gpt-4
``` 
