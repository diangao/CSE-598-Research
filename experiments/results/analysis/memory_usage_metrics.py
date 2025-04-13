import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_experiment_data, process_game_logs

def analyze_memory_call_frequency(df=None, save_path='experiments/results/analysis/figures'):
    """
    Analyze memory call frequency across different memory architectures and board sizes
    
    Args:
        df: Experiment data DataFrame, loaded if None
        save_path: Path to save figures
    """
    if df is None:
        df = load_experiment_data(experiment_type='constrained')
    
    os.makedirs(save_path, exist_ok=True)
    
    # Create memory call frequency analysis charts
    plt.figure(figsize=(15, 12))
    
    # 1. Total memory call frequency comparison (A vs B)
    plt.subplot(2, 2, 1)
    memory_calls_data = []
    
    for _, row in df.iterrows():
        memory_calls_data.append({
            'memory_constraint': row['memory_constraint_a'],
            'board_size': row['board_size'],
            'agent': 'Agent A',
            'total_memory_calls': row['agent_a_total_memory_calls']
        })
        memory_calls_data.append({
            'memory_constraint': row['memory_constraint_a'],
            'board_size': row['board_size'],
            'agent': 'Agent B',
            'total_memory_calls': row['agent_b_total_memory_calls']
        })
    
    memory_df = pd.DataFrame(memory_calls_data)
    sns.barplot(data=memory_df, x='memory_constraint', y='total_memory_calls', hue='agent')
    plt.title('Total Memory Calls by Memory Architecture and Agent')
    plt.xlabel('Memory Architecture')
    plt.ylabel('Total Memory Calls')
    
    # 2. Memory calls by type (Agent A)
    plt.subplot(2, 2, 2)
    memory_types_a = []
    
    for _, row in df.iterrows():
        memory_types_a.append({
            'memory_constraint': row['memory_constraint_a'],
            'board_size': row['board_size'],
            'memory_type': 'Graph',
            'calls': row['agent_a_graph_calls']
        })
        memory_types_a.append({
            'memory_constraint': row['memory_constraint_a'],
            'board_size': row['board_size'],
            'memory_type': 'Vector',
            'calls': row['agent_a_vector_calls']
        })
        memory_types_a.append({
            'memory_constraint': row['memory_constraint_a'],
            'board_size': row['board_size'],
            'memory_type': 'Semantic',
            'calls': row['agent_a_semantic_calls']
        })
    
    memory_types_a_df = pd.DataFrame(memory_types_a)
    sns.barplot(data=memory_types_a_df, x='memory_constraint', y='calls', hue='memory_type')
    plt.title('Agent A: Memory Calls by Type')
    plt.xlabel('Memory Architecture')
    plt.ylabel('Number of Calls')
    
    # 3. Memory calls by type (Agent B)
    plt.subplot(2, 2, 3)
    memory_types_b = []
    
    for _, row in df.iterrows():
        memory_types_b.append({
            'memory_constraint': row['memory_constraint_a'],
            'board_size': row['board_size'],
            'memory_type': 'Graph',
            'calls': row['agent_b_graph_calls']
        })
        memory_types_b.append({
            'memory_constraint': row['memory_constraint_a'],
            'board_size': row['board_size'],
            'memory_type': 'Vector',
            'calls': row['agent_b_vector_calls']
        })
        memory_types_b.append({
            'memory_constraint': row['memory_constraint_a'],
            'board_size': row['board_size'],
            'memory_type': 'Semantic',
            'calls': row['agent_b_semantic_calls']
        })
    
    memory_types_b_df = pd.DataFrame(memory_types_b)
    sns.barplot(data=memory_types_b_df, x='memory_constraint', y='calls', hue='memory_type')
    plt.title('Agent B: Memory Calls by Type')
    plt.xlabel('Memory Architecture')
    plt.ylabel('Number of Calls')
    
    # 4. Memory calls across board sizes
    plt.subplot(2, 2, 4)
    board_memory_impact = df.groupby(['memory_constraint_a', 'board_size']).agg({
        'agent_a_total_memory_calls': 'mean',
        'agent_b_total_memory_calls': 'mean'
    }).reset_index()
    
    for agent, marker in [('agent_a_total_memory_calls', 'o'), ('agent_b_total_memory_calls', 's')]:
        for constraint in sorted(board_memory_impact['memory_constraint_a'].unique()):
            data = board_memory_impact[board_memory_impact['memory_constraint_a'] == constraint]
            plt.plot(data['board_size'], data[agent], 
                     marker=marker, 
                     label=f"{agent.split('_')[0].title()} ({constraint})",
                     linestyle='-' if 'a' in agent else '--')
    
    plt.title('Memory Calls by Board Size and Memory Architecture')
    plt.xlabel('Board Size')
    plt.ylabel('Average Memory Calls')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'memory_call_frequency.png'))
    print(f"Memory call frequency analysis plot saved to {os.path.join(save_path, 'memory_call_frequency.png')}")
    
    # Generate detailed memory call statistics table
    memory_stats = df.groupby(['memory_constraint_a', 'board_size']).agg({
        'agent_a_graph_calls': ['mean', 'std', 'sum'],
        'agent_a_vector_calls': ['mean', 'std', 'sum'],
        'agent_a_semantic_calls': ['mean', 'std', 'sum'],
        'agent_a_total_memory_calls': ['mean', 'std', 'sum'],
        'agent_b_graph_calls': ['mean', 'std', 'sum'],
        'agent_b_vector_calls': ['mean', 'std', 'sum'],
        'agent_b_semantic_calls': ['mean', 'std', 'sum'],
        'agent_b_total_memory_calls': ['mean', 'std', 'sum']
    }).reset_index()
    
    # Save statistics to CSV
    memory_stats.to_csv(os.path.join(save_path, 'memory_call_statistics.csv'))
    print(f"Memory call statistics saved to {os.path.join(save_path, 'memory_call_statistics.csv')}")
    
    return memory_stats

def analyze_memory_operation_ratios(save_path='experiments/results/analysis/figures'):
    """
    Analyze memory operation ratios (store/read/schema update)
    
    Args:
        save_path: Path to save figures
    """
    # Load memory usage data from game logs
    memory_usage_df = process_game_logs(experiment_type='constrained')
    
    # If no memory usage data was extracted, return
    if memory_usage_df.empty:
        print("No memory usage data found in game logs.")
        return None
    
    os.makedirs(save_path, exist_ok=True)
    
    # Create memory operation ratio analysis charts
    plt.figure(figsize=(15, 12))
    
    # 1. Overall distribution of memory operations by type
    plt.subplot(2, 2, 1)
    operation_counts = memory_usage_df.groupby('operation').size().reset_index(name='count')
    
    if not operation_counts.empty:
        plt.pie(operation_counts['count'], labels=operation_counts['operation'], autopct='%1.1f%%')
        plt.title('Memory Operations Distribution')
    else:
        plt.text(0.5, 0.5, 'No operation data available', horizontalalignment='center', verticalalignment='center')
    
    # 2. Memory operations by agent and operation type
    plt.subplot(2, 2, 2)
    agent_operation = memory_usage_df.groupby(['agent_id', 'operation']).size().reset_index(name='count')
    
    if not agent_operation.empty:
        sns.barplot(data=agent_operation, x='agent_id', y='count', hue='operation')
        plt.title('Memory Operations by Agent')
        plt.xlabel('Agent')
        plt.ylabel('Number of Operations')
    else:
        plt.text(0.5, 0.5, 'No agent operation data available', horizontalalignment='center', verticalalignment='center')
    
    # 3. Memory operations by memory type and operation type
    plt.subplot(2, 2, 3)
    memory_operation = memory_usage_df.groupby(['memory_type', 'operation']).size().reset_index(name='count')
    
    if not memory_operation.empty:
        sns.barplot(data=memory_operation, x='memory_type', y='count', hue='operation')
        plt.title('Memory Operations by Memory Type')
        plt.xlabel('Memory Type')
        plt.ylabel('Number of Operations')
    else:
        plt.text(0.5, 0.5, 'No memory type operation data available', horizontalalignment='center', verticalalignment='center')
    
    # 4. Memory operations by constraint and operation type
    plt.subplot(2, 2, 4)
    constraint_operation = memory_usage_df.groupby(['memory_constraint_a', 'operation']).size().reset_index(name='count')
    
    if not constraint_operation.empty:
        sns.barplot(data=constraint_operation, x='memory_constraint_a', y='count', hue='operation')
        plt.title('Memory Operations by Memory Constraint')
        plt.xlabel('Memory Constraint')
        plt.ylabel('Number of Operations')
    else:
        plt.text(0.5, 0.5, 'No constraint operation data available', horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'memory_operation_ratios.png'))
    print(f"Memory operation ratios analysis plot saved to {os.path.join(save_path, 'memory_operation_ratios.png')}")
    
    # Generate detailed operation ratio statistics
    if not memory_usage_df.empty:
        # Calculate operation ratios for various combinations
        operation_stats = []
        
        # Calculate operation ratios by agent and memory constraint
        for (agent, constraint, memory_type), group in memory_usage_df.groupby(['agent_id', 'memory_constraint_a', 'memory_type']):
            total = len(group)
            if total > 0:
                # Calculate operation counts and ratios
                store_count = len(group[group['operation'] == 'store'])
                read_count = len(group[group['operation'] == 'read'])
                schema_count = len(group[group['operation'] == 'schema'])
                
                operation_stats.append({
                    'agent_id': agent,
                    'memory_constraint': constraint,
                    'memory_type': memory_type,
                    'total_operations': total,
                    'store_count': store_count,
                    'read_count': read_count,
                    'schema_count': schema_count,
                    'store_ratio': store_count / total if total > 0 else 0,
                    'read_ratio': read_count / total if total > 0 else 0,
                    'schema_ratio': schema_count / total if total > 0 else 0
                })
        
        if operation_stats:
            operation_stats_df = pd.DataFrame(operation_stats)
            operation_stats_df.to_csv(os.path.join(save_path, 'memory_operation_statistics.csv'))
            print(f"Memory operation statistics saved to {os.path.join(save_path, 'memory_operation_statistics.csv')}")
            return operation_stats_df
    
    return None

def analyze_game_phase_memory_usage(save_path='experiments/results/analysis/figures'):
    """
    Analyze memory usage patterns across different game phases (early/mid/late)
    
    Args:
        save_path: Path to save figures
    """
    # Load memory usage data from game logs
    memory_usage_df = process_game_logs(experiment_type='constrained')
    
    # If no memory usage data was extracted, return
    if memory_usage_df.empty:
        print("No memory usage data found in game logs.")
        return None
    
    os.makedirs(save_path, exist_ok=True)
    
    # Create game phase memory usage analysis charts
    plt.figure(figsize=(15, 12))
    
    # 1. Memory usage distribution by game phase
    plt.subplot(2, 2, 1)
    phase_counts = memory_usage_df.groupby('phase').size().reset_index(name='count')
    
    if not phase_counts.empty:
        sns.barplot(data=phase_counts, x='phase', y='count')
        plt.title('Memory Usage by Game Phase')
        plt.xlabel('Game Phase')
        plt.ylabel('Number of Memory Calls')
    else:
        plt.text(0.5, 0.5, 'No phase data available', horizontalalignment='center', verticalalignment='center')
    
    # 2. Memory usage by game phase and agent
    plt.subplot(2, 2, 2)
    agent_phase = memory_usage_df.groupby(['agent_id', 'phase']).size().reset_index(name='count')
    
    if not agent_phase.empty:
        sns.barplot(data=agent_phase, x='phase', y='count', hue='agent_id')
        plt.title('Memory Usage by Game Phase and Agent')
        plt.xlabel('Game Phase')
        plt.ylabel('Number of Memory Calls')
    else:
        plt.text(0.5, 0.5, 'No agent phase data available', horizontalalignment='center', verticalalignment='center')
    
    # 3. Memory types used by game phase
    plt.subplot(2, 2, 3)
    phase_memory_type = memory_usage_df.groupby(['phase', 'memory_type']).size().reset_index(name='count')
    
    if not phase_memory_type.empty:
        sns.barplot(data=phase_memory_type, x='phase', y='count', hue='memory_type')
        plt.title('Memory Types Used by Game Phase')
        plt.xlabel('Game Phase')
        plt.ylabel('Number of Memory Calls')
    else:
        plt.text(0.5, 0.5, 'No phase memory type data available', horizontalalignment='center', verticalalignment='center')
    
    # 4. Memory operations by game phase
    plt.subplot(2, 2, 4)
    phase_operation = memory_usage_df.groupby(['phase', 'operation']).size().reset_index(name='count')
    
    if not phase_operation.empty:
        sns.barplot(data=phase_operation, x='phase', y='count', hue='operation')
        plt.title('Memory Operations by Game Phase')
        plt.xlabel('Game Phase')
        plt.ylabel('Number of Operations')
    else:
        plt.text(0.5, 0.5, 'No phase operation data available', horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'game_phase_memory_usage.png'))
    print(f"Game phase memory usage analysis plot saved to {os.path.join(save_path, 'game_phase_memory_usage.png')}")
    
    # Calculate game phase usage statistics
    if not memory_usage_df.empty:
        # Statistics by game phase, agent and memory type
        phase_stats = []
        
        for (phase, agent, memory_type), group in memory_usage_df.groupby(['phase', 'agent_id', 'memory_type']):
            store_count = len(group[group['operation'] == 'store'])
            read_count = len(group[group['operation'] == 'read'])
            schema_count = len(group[group['operation'] == 'schema'])
            total = len(group)
            
            # Calculate win rate
            if 'agent_won' in group.columns:
                wins = group['agent_won'].sum()
                win_rate = wins / len(group) if len(group) > 0 else 0
            else:
                wins = 0
                win_rate = 0
            
            phase_stats.append({
                'phase': phase,
                'agent_id': agent,
                'memory_type': memory_type,
                'total_calls': total,
                'store_count': store_count,
                'read_count': read_count,
                'schema_count': schema_count,
                'store_ratio': store_count / total if total > 0 else 0,
                'read_ratio': read_count / total if total > 0 else 0,
                'schema_ratio': schema_count / total if total > 0 else 0,
                'wins': wins,
                'win_rate': win_rate
            })
        
        if phase_stats:
            phase_stats_df = pd.DataFrame(phase_stats)
            phase_stats_df.to_csv(os.path.join(save_path, 'game_phase_statistics.csv'))
            print(f"Game phase statistics saved to {os.path.join(save_path, 'game_phase_statistics.csv')}")
            return phase_stats_df
    
    return None

def analyze_memory_and_win_relationship(df=None, memory_usage_df=None, save_path='experiments/results/analysis/figures'):
    """
    Analyze the relationship between memory usage and winning
    
    Args:
        df: Experiment data DataFrame, loaded if None
        memory_usage_df: Memory usage data DataFrame, loaded if None
        save_path: Path to save figures
    """
    if df is None:
        df = load_experiment_data(experiment_type='constrained')
    
    if memory_usage_df is None:
        memory_usage_df = process_game_logs(experiment_type='constrained')
    
    os.makedirs(save_path, exist_ok=True)
    
    # Create memory usage and winning relationship analysis charts
    plt.figure(figsize=(15, 10))
    
    # 1. Relationship between total memory calls and win rate
    plt.subplot(2, 2, 1)
    relation_data = []
    
    for _, row in df.iterrows():
        relation_data.append({
            'memory_constraint': row['memory_constraint_a'],
            'agent': 'Agent A',
            'memory_calls': row['agent_a_total_memory_calls'],
            'win_rate': row['agent_a_win_rate']
        })
        relation_data.append({
            'memory_constraint': row['memory_constraint_a'],
            'agent': 'Agent B',
            'memory_calls': row['agent_b_total_memory_calls'],
            'win_rate': row['agent_b_win_rate']
        })
    
    relation_df = pd.DataFrame(relation_data)
    sns.scatterplot(data=relation_df, x='memory_calls', y='win_rate', hue='agent', style='memory_constraint', s=100)
    plt.title('Win Rate vs Memory Calls')
    plt.xlabel('Total Memory Calls')
    plt.ylabel('Win Rate')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 2. Memory usage comparison between winners and non-winners
    plt.subplot(2, 2, 2)
    
    if not memory_usage_df.empty and 'agent_won' in memory_usage_df.columns:
        win_memory = memory_usage_df.groupby(['agent_id', 'agent_won']).size().reset_index(name='memory_calls')
        
        if not win_memory.empty:
            sns.barplot(data=win_memory, x='agent_id', y='memory_calls', hue='agent_won')
            plt.title('Memory Usage by Winners vs Non-Winners')
            plt.xlabel('Agent')
            plt.ylabel('Number of Memory Calls')
            plt.legend(title='Agent Won')
        else:
            plt.text(0.5, 0.5, 'No winner data available', horizontalalignment='center', verticalalignment='center')
    else:
        plt.text(0.5, 0.5, 'No agent_won data available', horizontalalignment='center', verticalalignment='center')
    
    # 3. Relationship between memory type usage and win rate (Agent A)
    plt.subplot(2, 2, 3)
    memory_type_win_a = []
    
    for _, row in df.iterrows():
        if row['agent_a_total_memory_calls'] > 0:
            memory_type_win_a.append({
                'memory_constraint': row['memory_constraint_a'],
                'memory_type': 'Graph',
                'usage_ratio': row['agent_a_graph_calls'] / row['agent_a_total_memory_calls'] if row['agent_a_total_memory_calls'] > 0 else 0,
                'win_rate': row['agent_a_win_rate']
            })
            memory_type_win_a.append({
                'memory_constraint': row['memory_constraint_a'],
                'memory_type': 'Vector',
                'usage_ratio': row['agent_a_vector_calls'] / row['agent_a_total_memory_calls'] if row['agent_a_total_memory_calls'] > 0 else 0,
                'win_rate': row['agent_a_win_rate']
            })
    
    memory_type_win_a_df = pd.DataFrame(memory_type_win_a)
    
    if not memory_type_win_a_df.empty:
        sns.scatterplot(data=memory_type_win_a_df, x='usage_ratio', y='win_rate', hue='memory_type', style='memory_constraint', s=100)
        plt.title('Agent A: Memory Type Usage Ratio vs Win Rate')
        plt.xlabel('Memory Type Usage Ratio')
        plt.ylabel('Win Rate')
        plt.grid(True, linestyle='--', alpha=0.5)
    else:
        plt.text(0.5, 0.5, 'Insufficient data for Agent A', horizontalalignment='center', verticalalignment='center')
    
    # 4. Relationship between memory type usage and win rate (Agent B)
    plt.subplot(2, 2, 4)
    memory_type_win_b = []
    
    for _, row in df.iterrows():
        if row['agent_b_total_memory_calls'] > 0:
            memory_type_win_b.append({
                'memory_constraint': row['memory_constraint_a'],
                'memory_type': 'Graph',
                'usage_ratio': row['agent_b_graph_calls'] / row['agent_b_total_memory_calls'] if row['agent_b_total_memory_calls'] > 0 else 0,
                'win_rate': row['agent_b_win_rate']
            })
            memory_type_win_b.append({
                'memory_constraint': row['memory_constraint_a'],
                'memory_type': 'Vector',
                'usage_ratio': row['agent_b_vector_calls'] / row['agent_b_total_memory_calls'] if row['agent_b_total_memory_calls'] > 0 else 0,
                'win_rate': row['agent_b_win_rate']
            })
    
    memory_type_win_b_df = pd.DataFrame(memory_type_win_b)
    
    if not memory_type_win_b_df.empty:
        sns.scatterplot(data=memory_type_win_b_df, x='usage_ratio', y='win_rate', hue='memory_type', style='memory_constraint', s=100)
        plt.title('Agent B: Memory Type Usage Ratio vs Win Rate')
        plt.xlabel('Memory Type Usage Ratio')
        plt.ylabel('Win Rate')
        plt.grid(True, linestyle='--', alpha=0.5)
    else:
        plt.text(0.5, 0.5, 'Insufficient data for Agent B', horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'memory_win_relationship.png'))
    print(f"Memory and win relationship analysis plot saved to {os.path.join(save_path, 'memory_win_relationship.png')}")
    
    return relation_df

if __name__ == "__main__":
    # Load data
    df = load_experiment_data(experiment_type='constrained')
    memory_usage_df = process_game_logs(experiment_type='constrained')
    
    # Create directory for saving figures
    os.makedirs('experiments/results/analysis/figures', exist_ok=True)
    
    # Run all analyses
    analyze_memory_call_frequency(df)
    analyze_memory_operation_ratios()
    analyze_game_phase_memory_usage()
    analyze_memory_and_win_relationship(df, memory_usage_df)
    
    print("Memory usage metrics analysis completed.") 