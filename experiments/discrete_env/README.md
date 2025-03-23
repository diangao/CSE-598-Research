
# Discrete Task Implementation Guide (Internal)

## Core Scientific Questions
1. Does memory architecture affect planning depth? (τ-bench)
2. How does representation type impact move efficiency?
3. What's the failure mode when transferring between grid sizes?

## Phase 1: Controlled Baseline (Mar 9-12)
### Implementation Checklist
```python
# Validation test suite
def test_representation_integrity():
    """Verify state encoding consistency"""
    for memory_type in ['graph', 'vector', 'semantic']:
        agent = load_agent(memory_type)
        original_state = env.reset()
        encoded = agent.encode(original_state)
        reconstructed = agent.decode(encoded)
        
        # Quantitative measurement
        assert np.allclose(original_state, reconstructed, atol=0.1), \
            f"{memory_type} representation failed reconstruction"
```

### Critical Parameters
| Parameter         | GraphDB      | VectorDB     | Semantic     |
|-------------------|--------------|--------------|--------------|
| State Dimension   | 9 (raw)      | 64 (latent)  | 384 (BERT)   |
| Batch Size        | N/A          | 32           | 16           |
| Update Frequency  | Per move     | Every 5 moves| End of game  |

## Phase 2: Strategic Depth Analysis (Mar 13-17)
### τ-bench Implementation
```python
def measure_planning_depth(agent, env, max_depth=5):
    """Quantify multi-step reasoning capacity"""
    depth_scores = []
    for depth in range(1, max_depth+1):
        wins = 0
        for _ in range(100):
            # Force lookahead depth
            agent.memory.set_search_depth(depth)  
            result = env.run_episode(agent)
            wins += int(result == 'win')
        
        # Calculate depth efficiency
        efficiency = wins / 100
        depth_scores.append({
            'depth': depth,
            'efficiency': efficiency,
            'agent_type': type(agent).__name__
        })
    
    return depth_scores
```

### Validation Protocol
1. **Controlled Opposition**
   - Perfect opponent: Minimax algorithm with depth=4
   - Stochastic opponent: 70% optimal moves
   
2. **Perturbation Tests**
   ```bash
   # Induce controlled failures
   python run_perturbation.py --noise_type=spatial --intensity=0.3
   python run_perturbation.py --noise_type=temporal --interval=5
   ```

## Phase 3: Cross-Architecture Validation (Mar 18-22)
### Statistical Comparison Matrix
| Comparison            | Hypothesis                  | Test Method       |
|-----------------------|-----------------------------|-------------------|
| Graph vs Vector       | Explicit > Implicit planning| Paired t-test     |
| Vector vs Semantic    | Latent space stability      | KL divergence     |
| All architectures     | Move efficiency             | ANOVA             |

### Effect Size Measurement
```python
def calculate_cohens_d(group1, group2):
    """Quantify practical significance"""
    diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt((np.std(group1)**2 + np.std(group2)**2)/2)
    return abs(diff / pooled_std)
```

## Failure Mode Analysis
### Expected Issues & Solutions
1. **Memory Leak in GraphDB**
   - Symptom: O(n) memory growth
   - Fix: Implement node pruning (keep depth ≤ 5)

2. **VectorDB Collapse**
   - Detection: ‖embedding‖ < 0.1
   - Recovery: Re-initialize encoder layers 3-6

3. **Semantic Drift**
   - Monitor: BERTScore between turns
   - Threshold: < 0.7 triggers memory refresh

## Data Collection Standard
```python
# Required fields per experiment run
DATA_SCHEMA = {
    'timestamp': 'ISO 8601',
    'agent_config': 'SHA256 hash',
    'environment_params': {
        'grid_size': 'int',
        'opponent_type': 'str'
    },
    'metrics': {
        'win_rate': 'float',
        'move_efficiency': 'float',
        'planning_depth': 'int',
        'memory_usage': 'MB'
    }
}
```


2. **Performance Variance**
   - Check temperature settings in VectorDB
   - Verify graph pruning is active
   - Monitor LLM attention patterns in Semantic