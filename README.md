# CSE 598: Science of LLMs Research Project

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Systematic comparison of state representation architectures for multi-agent systems, analyzing GraphDB, VectorDB, and Semantic Memory approaches.

## 📂 Project Structure

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
├── data/                     # Preprocessed datasets
├── docs/                     # Documentation
└── requirements.txt          # Python dependencies
```

## 🚀 Quick Start

1. **Install dependencies**:
bash
pip install -r requirements.txt

2. **Run experiments**:
TBD

3. **View results**:
TBD

## 🔬 Key Experiments

| Experiment | Description | Metrics |
|------------|-------------|---------|
| Discrete Task Analysis | Compare memory architectures in 3x3 Tic-Tac-Toe | Win Rate, Move Efficiency |
| Continuous Adaptation | Evaluate latent-space projections | KL Divergence, Recovery Rate |
| Cross-Arch Transfer | Test knowledge transfer between memory systems | Adaptation Speed, Stability Index |

## 📊 Expected Outcomes

- Comparative analysis of memory architectures
- Cross-environment generalization metrics
- Parameter-efficient optimization benchmarks

## 📅 Development Timeline

- **Mar 1-15**: Discrete task implementation
- **Mar 16-31**: Continuous environment setup
- **Apr 1-15**: Cross-architecture analysis
- **Apr 16-30**: Final report preparation

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.