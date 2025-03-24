# CSE 598 - Science of LLMs Research: State Representation Learning for Long-Term Multi-Agent Interactions

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A research project exploring different memory architectures for multi-agent systems, comparing GraphDB, VectorDB, and Semantic Memory approaches in both discrete and continuous environments.

## Latest Progress

### Tic-Tac-Toe Memory Architecture Experiment
We've successfully implemented our first experiment: comparing different memory architectures in 3x3 Tic-Tac-Toe games through integration with τ-Bench framework.

**Key Components Implemented:**
- Custom TicTacToe discrete environment with reward mechanism
- Three memory architectures with unified interface:
  - **GraphMemory**: State-action relationships stored as graph structures
  - **VectorMemory**: Board state embeddings with vector similarity search
  - **SemanticMemory**: Natural language descriptions with semantic retrieval
- Memory-augmented agents leveraging ReAct reasoning framework
- Comprehensive experiment runner with metrics collection

**Metrics Being Measured:**
- Win rate across different memory architectures
- Retrieval latency for each memory type
- Memory usage patterns and efficiency
- Move accuracy influenced by memory retrieval

To run the experiment:
```bash
cd integration/tau-bench
python run_tictactoe_experiment.py --model gpt-3.5-turbo --model-provider openai --num-episodes 100
```

Results are saved in `results/tictactoe_experiment/` with detailed logs and summary statistics.

## Research Focus
- Investigating optimal state representation methods for LLM-powered agents
- Comparing different memory architectures in multi-agent interactions
- Evaluating cross-domain knowledge transfer capabilities

## Project Structure

```
CSE-598-Research/
├── proposal/                  # LaTeX research proposal
│   ├── proposal.tex          # Main document
│   └── proposal.pdf          # PDF version
├── experiments/              # Core implementation
│   ├── discrete_env/         # 3x3 Tic-Tac-Toe experiments
│   │   ├── agents/          # Agent implementations
│   │   └── analysis/        # Performance metrics
│   ├── continuous_env/       # Latent-space experiments
│   └── utils/               # Common utilities
├── integration/              # Framework integrations
│   └── tau-bench/           # τ-Bench integration
│       ├── tau_bench/       # Extended τ-Bench implementation
│       │   ├── memory/      # Custom memory modules
│       │   ├── envs/        # Custom environments
│       │   └── agents/      # Memory-augmented agents
│       └── run_tictactoe_experiment.py  # Experiment runner
├── data/                     # Preprocessed datasets
├── docs/                     # Documentation
└── requirements.txt          # Python dependencies
```

## Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run experiments**:
```bash
# Run Tic-Tac-Toe experiment with all memory types
cd integration/tau-bench
python run_tictactoe_experiment.py --model gpt-3.5-turbo --num-episodes 30

# Or test with specific memory type
python run_tictactoe_experiment.py --agents graph --num-episodes 10
```

3. **View results**:
Results are saved in `integration/tau-bench/results/tictactoe_experiment/` including:
- Detailed game logs for each agent
- Performance metrics and statistics
- Memory usage patterns

## Key Experiments

| Experiment | Description | Metrics | Status |
|------------|-------------|---------|--------|
| Discrete Task Analysis | Compare memory architectures in 3x3 Tic-Tac-Toe | Win Rate, Move Efficiency, Retrieval Latency | ✅ Implemented |
| Continuous Adaptation | Evaluate latent-space projections | KL Divergence, Recovery Rate | 🔄 Planned |
| Cross-Arch Transfer | Test knowledge transfer between memory systems | Adaptation Speed, Stability Index | 🔄 Planned |

## Technologies
- Memory Systems: GraphDB, VectorDB, Semantic Memory
- LLM Integration: OpenAI GPT-3.5/GPT-4, DeepSeek
- Evaluation Framework: τ-bench based metrics
- Architecture: ReAct reasoning framework with memory augmentation

## Environment Setup Guide

### Setting Up the Environment

```bash
# Clone the repository
git clone https://github.com/diangao/CSE-598-Research.git
cd CSE-598-Research

# Create a virtual environment (recommended)
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Resolving Dependency Conflicts

We've observed some package compatibility issues during development. Here's how to resolve them:

1. **OpenAI and LiteLLM Version Compatibility**:
   ```bash
   # If you encounter errors about "CompletionTokensDetails"
   pip uninstall -y openai litellm
   pip install "openai>=1.5.0" "litellm>=1.16.0"
   ```

2. **Sentence Transformers (optional)**:
   For semantic memory to work with optimized embeddings:
   ```bash
   pip install sentence-transformers
   ```

3. **NetworkX and NumPy Versions**:
   If you encounter NetworkX compatibility errors:
   ```bash
   pip install "networkx>=3.1" "numpy>=1.24.3"
   ```

### API Key Setup

You need to set up an API key to run the experiments with LLM providers:

```bash
# For OpenAI (required for default configuration)
export OPENAI_API_KEY="your-openai-api-key-here"

# For DeepSeek (optional alternative)
export DEEPSEEK_API_KEY="your-deepseek-api-key-here"
```

On Windows, use:
```bash
set OPENAI_API_KEY=your-openai-api-key-here
```

##  Running Experiments

### Tic-Tac-Toe Memory Experiments

Navigate to the integration directory:
```bash
cd integration/tau-bench
```

#### Basic Experiments
```bash
# Run with all memory types (10 episodes each)
python run_tictactoe_experiment.py --num-episodes 10

# Test only with graph memory
python run_tictactoe_experiment.py --agents graph --num-episodes 10

# Test with multiple memory types
python run_tictactoe_experiment.py --agents graph,vector --num-episodes 10
```

#### Advanced Configuration
```bash
# Specify a different model and provider
python run_tictactoe_experiment.py --model gpt-4o --model-provider openai --num-episodes 10

# Adjust temperature for more exploration
python run_tictactoe_experiment.py --temperature 0.7 --num-episodes 10

# Set a specific random seed for reproducibility
python run_tictactoe_experiment.py --seed 123 --num-episodes 10
```

### Viewing Results

Results are stored in the `results/tictactoe_experiment` directory:
- `logs/`: Contains detailed logs for each experiment run
- `summaries/`: Contains summary statistics in JSON format

### Interpreting Results

Our experimental results capture multiple dimensions of memory system performance:

#### 1. Win Rate Analysis
Win rates directly measure decision quality influenced by memory:
- **High win rate (>60%)**: Indicates effective memory retrieval and utilization
- **Similar win rates across memory types**: Suggests the task may not sufficiently differentiate memory architectures
- **Win rate differences >10%**: Statistically significant indicator of memory architecture advantages

#### 2. Memory Performance Metrics
Each result contains detailed memory performance statistics:
```json
"memory_stats": {
  "retrieval_times": [0.00031, 0.00029, ...],
  "storage_count": 27
}
```

Key interpretation points:
- **Retrieval times**: Lower average retrieval time indicates more efficient memory architecture
  - GraphMemory: Typically fastest for small state spaces
  - VectorMemory: Balance of speed and semantic richness
  - SemanticMemory: Slowest but potentially most contextually relevant

- **Storage count**: Higher numbers indicate more state exploration
  - Low storage (<10 per game): May indicate repetitive play patterns
  - High variance between games: Shows exploration inconsistency

#### 3. Move Analysis
Examine game traces to understand memory influence on decision quality:
- Look for repeated mistakes despite similar board positions in memory
- Identify instances where memory retrieval improved decisions
- Check for memory utilization increases across successive games

#### 4. Visualizing Comparisons
For comparative analysis across memory types:
```bash
# Example Python snippet for visualizing results
import json
import matplotlib.pyplot as plt

# Load summary files
with open('results/tictactoe_experiment/summaries/experiment_summary.json', 'r') as f:
    data = json.load(f)

# Extract win rates
memory_types = ['graph', 'vector', 'semantic']
win_rates = [sum(1 for r in data['results'][m] if r['reward'] > 0.5)/len(data['results'][m]) 
             for m in memory_types]

# Plot comparison
plt.bar(memory_types, win_rates)
plt.title('Win Rate by Memory Architecture')
plt.ylabel('Win Rate')
plt.savefig('memory_comparison.png')
```

#### 5. Common Patterns to Look For
- **Graph memory advantages**: Typically shows strengths in highly structured games with clear state transitions
- **Vector memory balance**: Often provides best compromise between retrieval speed and representation quality
- **Semantic memory context**: May perform better when game state has complex semantic meaning
- **Memory evolution**: Performance should improve as more games are played and memory builds up

For detailed analysis tutorials, see the Jupyter notebooks in `experiments/discrete_env/analysis/`.

## Expected Outcomes
- Comparative analysis of memory architectures
- Cross-environment generalization metrics
- Parameter-efficient optimization benchmarks

## Development Timeline

- **Mar 1-15**: Discrete task implementation ✅
- **Mar 16-31**: Continuous environment setup 🔄
- **Apr 1-15**: Cross-architecture analysis 🔄
- **Apr 16-30**: Final report preparation 🔄

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
