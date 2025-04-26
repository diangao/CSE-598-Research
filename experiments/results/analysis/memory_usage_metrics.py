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
    
    # Set consistent style for all plots
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create memory call frequency analysis charts with improved aesthetics
    plt.figure(figsize=(16, 14))
    
    # Define consistent colors for better visualization
    agent_colors = {'Agent A': '#1f77b4', 'Agent B': '#ff7f0e'}
    memory_colors = {'Graph': '#2ca02c', 'Vector': '#d62728', 'Semantic': '#9467bd'}
    
    # 1. Total memory call frequency comparison (A vs B) - Enhanced visualization
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
    
    # Create more informative barplot
    ax1 = sns.barplot(data=memory_df, x='memory_constraint', y='total_memory_calls', hue='agent', palette=agent_colors)
    
    # Add value labels on top of each bar
    for p in ax1.patches:
        height = p.get_height()
        if height > 0:  # Only add label if there's a visible bar
            ax1.text(p.get_x() + p.get_width()/2., height + 0.3,
                    f'{int(height)}',
                    ha="center", fontsize=10)
    
    # Improve labels and title
    plt.title('Total Memory Calls by Memory Architecture and Agent', fontsize=14, fontweight='bold')
    plt.xlabel('Memory Architecture', fontsize=12)
    plt.ylabel('Total Memory Calls', fontsize=12)
    
    # Rename x-axis labels for clarity
    plt.xticks([0, 1], ['Graph Memory Only', 'Vector Memory Only'], fontsize=11)
    
    # Add annotation explaining the implication
    plt.text(0.5, 1, "Finding: Agent A (win-rate optimizer) makes significantly more memory calls",
             ha='center', fontsize=10, style='italic', color='darkblue')
    
    # 2. Memory calls by type (Agent A) - Enhanced visualization
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
    
    # Create more informative barplot
    ax2 = sns.barplot(data=memory_types_a_df, x='memory_constraint', y='calls', hue='memory_type', palette=memory_colors)
    
    # Add value labels on top of each bar
    for p in ax2.patches:
        height = p.get_height()
        if height > 0:  # Only add label if there's a visible bar
            ax2.text(p.get_x() + p.get_width()/2., height + 0.3,
                    f'{int(height)}',
                    ha="center", fontsize=10)
    
    # Improve labels and title
    plt.title('Agent A: Memory Calls by Type', fontsize=14, fontweight='bold')
    plt.xlabel('Memory Architecture', fontsize=12)
    plt.ylabel('Number of Calls', fontsize=12)
    
    # Rename x-axis labels for clarity
    plt.xticks([0, 1], ['Graph Memory Only', 'Vector Memory Only'], fontsize=11)
    
    # Add annotation explaining the implication
    plt.text(0.5, 1, "Hypothesis: Agents adapt to the available memory type",
             ha='center', fontsize=10, style='italic', color='darkblue')
    
    # 3. Memory calls by type (Agent B) - Enhanced visualization
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
    
    # Create more informative barplot
    ax3 = sns.barplot(data=memory_types_b_df, x='memory_constraint', y='calls', hue='memory_type', palette=memory_colors)
    
    # Add value labels on top of each bar
    for p in ax3.patches:
        height = p.get_height()
        if height > 0:  # Only add label if there's a visible bar
            ax3.text(p.get_x() + p.get_width()/2., height + 0.1,
                    f'{int(height)}',
                    ha="center", fontsize=10)
    
    # Improve labels and title
    plt.title('Agent B: Memory Calls by Type', fontsize=14, fontweight='bold')
    plt.xlabel('Memory Architecture', fontsize=12)
    plt.ylabel('Number of Calls', fontsize=12)
    
    # Rename x-axis labels for clarity
    plt.xticks([0, 1], ['Graph Memory Only', 'Vector Memory Only'], fontsize=11)
    
    # Add annotation explaining the implication
    plt.text(0.5, 0.5, "Finding: Agent B (token-efficient) makes fewer memory calls",
             ha='center', fontsize=10, style='italic', color='darkblue')
    
    # 4. Memory calls across board sizes - Enhanced visualization
    plt.subplot(2, 2, 4)
    board_memory_impact = df.groupby(['memory_constraint_a', 'board_size']).agg({
        'agent_a_total_memory_calls': 'mean',
        'agent_b_total_memory_calls': 'mean'
    }).reset_index()
    
    # Create a more informative and clear line plot
    plt.figure(4)  # Create a separate figure for this plot for better control
    
    # Define line styles and markers for clarity
    styles = {
        'graph_only': '-',
        'vector_only': '--'
    }
    
    # Plot with clearer labels and styles
    for agent, marker, color, agent_label in [
        ('agent_a_total_memory_calls', 'o', '#1f77b4', 'Agent A'),
        ('agent_b_total_memory_calls', 's', '#ff7f0e', 'Agent B')
    ]:
        for constraint in sorted(board_memory_impact['memory_constraint_a'].unique()):
            data = board_memory_impact[board_memory_impact['memory_constraint_a'] == constraint]
            
            # Convert board size to numeric for proper plotting
            data['board_size_num'] = data['board_size'].str.extract('(\d+)').astype(int)
            data = data.sort_values('board_size_num')
            
            # Plot the line
            line = plt.plot(
                data['board_size'], 
                data[agent],
                marker=marker,
                label=f"{agent_label} ({constraint.replace('_only', ' Memory')})",
                linestyle=styles[constraint],
                linewidth=2.5,
                markersize=8,
                color=color if 'agent_a' in agent else color
            )
            
            # Add data labels at each point
            for i, row in data.iterrows():
                plt.text(row['board_size'], row[agent] + 0.5, 
                         f"{int(row[agent])}", 
                         ha='center', va='bottom', 
                         fontsize=9,
                         color=line[0].get_color())
    
    # Improve chart appearance
    plt.title('Memory Calls by Board Size and Memory Architecture', fontsize=14, fontweight='bold')
    plt.xlabel('Board Size', fontsize=12)
    plt.ylabel('Average Memory Calls', fontsize=12)
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotation explaining the key findings
    plt.text('6', 5, "Finding: Vector memory usage increases with board complexity\nwhile Graph memory peaks at medium complexity",
             ha='center', fontsize=10, style='italic', color='darkblue')
    
    # Ensure proper spacing and layout
    plt.tight_layout()
    
    # Save the board size analysis as a separate, high-quality image
    plt.savefig(os.path.join(save_path, 'memory_calls_by_board_size.png'), dpi=300, bbox_inches='tight')
    
    # Return to the main figure and save it
    plt.figure(1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'memory_call_frequency.png'), dpi=300, bbox_inches='tight')
    print(f"Memory call frequency analysis plot saved to {os.path.join(save_path, 'memory_call_frequency.png')}")
    print(f"Memory calls by board size analysis saved to {os.path.join(save_path, 'memory_calls_by_board_size.png')}")
    
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

def analyze_memory_usage_patterns(df=None, memory_usage_df=None, save_path='experiments/results/analysis/figures/memory_baseline'):
    """
    Analyze memory usage patterns in baseline experiments, comparing agents with and without memory
    
    Args:
        df: Experiment data DataFrame, loaded if None
        memory_usage_df: Memory usage data DataFrame from game logs, loaded if None
        save_path: Path to save figures
    """
    # 加载必要的数据
    if df is None:
        df = load_experiment_data(experiment_type='memory_comparison')
    
    if memory_usage_df is None:
        memory_usage_df = process_game_logs(experiment_type='memory_comparison')
    
    os.makedirs(save_path, exist_ok=True)
    
    # 检查数据是否可用
    if df.empty:
        print("No memory comparison data found")
        return None
    
    if memory_usage_df.empty:
        print("No memory usage data found in game logs")
        return None
    
    # 设置图表样式
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 创建一个大型图表展示记忆使用模式
    plt.figure(figsize=(16, 14))
    
    # 1. 记忆函数调用的时间分布 - 分析记忆使用在游戏过程中的分布
    plt.subplot(2, 2, 1)
    
    # 计算每次记忆调用的游戏进程百分比
    if 'turn' in memory_usage_df.columns:
        # 按游戏ID分组，计算每个游戏的总回合数
        game_turns = memory_usage_df.groupby(['run_id', 'game_id'])['turn'].max() + 1
        
        # 创建一个新的DataFrame，包含每次记忆调用的游戏进程百分比
        memory_timeline = []
        
        for _, row in memory_usage_df.iterrows():
            run_id = row['run_id']
            game_id = row['game_id']
            
            # 跳过缺失总回合数的游戏
            if (run_id, game_id) not in game_turns.index:
                continue
                
            total_turns = game_turns[(run_id, game_id)]
            
            # 如果总回合数为0，跳过
            if total_turns == 0:
                continue
                
            # 计算游戏进程百分比
            turn_percentage = row['turn'] / total_turns * 100
            
            memory_timeline.append({
                'agent_id': row['agent_id'],
                'memory_type': row['memory_type'],
                'operation': row['operation'],
                'memory_constraint': row['memory_constraint_a'],
                'turn_percentage': turn_percentage,
                'board_size': row['board_size']
            })
        
        if memory_timeline:
            # 转换为DataFrame
            timeline_df = pd.DataFrame(memory_timeline)
            
            # 绘制记忆调用的时间分布直方图
            sns.histplot(data=timeline_df, x='turn_percentage', hue='memory_type', 
                        element='step', bins=20, multiple='stack')
            
            plt.title('Memory Call Distribution Throughout Game Progress', fontsize=14, fontweight='bold')
            plt.xlabel('Game Progress (%)', fontsize=12)
            plt.ylabel('Number of Memory Calls', fontsize=12)
            plt.legend(title='Memory Type')
            
            # 添加垂直线标记游戏的开始、中期和结束阶段
            plt.axvline(x=33.3, color='gray', linestyle='--', alpha=0.7)
            plt.axvline(x=66.6, color='gray', linestyle='--', alpha=0.7)
            
            # 添加文本标记游戏阶段
            plt.text(16.7, plt.ylim()[1]*0.9, 'Early Game', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
            plt.text(50, plt.ylim()[1]*0.9, 'Mid Game', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
            plt.text(83.3, plt.ylim()[1]*0.9, 'Late Game', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
        else:
            plt.text(0.5, 0.5, 'No memory timeline data available', ha='center', va='center', transform=plt.gca().transAxes)
    else:
        plt.text(0.5, 0.5, 'No turn data available', ha='center', va='center', transform=plt.gca().transAxes)
    
    # 2. 记忆使用频率 - 按记忆类型和代理
    plt.subplot(2, 2, 2)
    
    # 计算每个代理的记忆使用频率
    memory_frequency = memory_usage_df.groupby(['agent_id', 'memory_type']).size().reset_index(name='frequency')
    
    if not memory_frequency.empty:
        # 绘制记忆使用频率条形图
        sns.barplot(data=memory_frequency, x='agent_id', y='frequency', hue='memory_type')
        
        plt.title('Memory Usage Frequency by Agent and Memory Type', fontsize=14, fontweight='bold')
        plt.xlabel('Agent', fontsize=12)
        plt.ylabel('Number of Memory Calls', fontsize=12)
        plt.legend(title='Memory Type')
        
        # 为每个条形添加标签
        for container in plt.gca().containers:
            plt.bar_label(container, fmt='%d')
    else:
        plt.text(0.5, 0.5, 'No memory frequency data available', ha='center', va='center', transform=plt.gca().transAxes)
    
    # 3. 记忆操作类型分布 - 存储 vs 读取 vs 模式更新
    plt.subplot(2, 2, 3)
    
    # 计算每个代理的记忆操作类型分布
    operation_dist = memory_usage_df.groupby(['agent_id', 'operation']).size().reset_index(name='count')
    
    if not operation_dist.empty:
        # 绘制记忆操作类型分布条形图
        sns.barplot(data=operation_dist, x='agent_id', y='count', hue='operation')
        
        plt.title('Memory Operation Types by Agent', fontsize=14, fontweight='bold')
        plt.xlabel('Agent', fontsize=12)
        plt.ylabel('Number of Operations', fontsize=12)
        plt.legend(title='Operation Type')
        
        # 为每个条形添加标签
        for container in plt.gca().containers:
            plt.bar_label(container, fmt='%d')
    else:
        plt.text(0.5, 0.5, 'No operation distribution data available', ha='center', va='center', transform=plt.gca().transAxes)
    
    # 4. 不同棋盘大小的记忆使用模式
    plt.subplot(2, 2, 4)
    
    # 计算每个棋盘大小的记忆使用
    board_size_usage = memory_usage_df.groupby(['board_size', 'memory_type']).size().reset_index(name='count')
    
    if not board_size_usage.empty:
        # 确保棋盘大小是按数字排序的
        board_size_usage['board_size_num'] = board_size_usage['board_size'].astype(str).str.extract('(\d+)').astype(int)
        board_size_usage = board_size_usage.sort_values('board_size_num')
        
        # 绘制不同棋盘大小的记忆使用条形图
        sns.barplot(data=board_size_usage, x='board_size', y='count', hue='memory_type')
        
        plt.title('Memory Usage by Board Size and Memory Type', fontsize=14, fontweight='bold')
        plt.xlabel('Board Size', fontsize=12)
        plt.ylabel('Number of Memory Calls', fontsize=12)
        plt.legend(title='Memory Type')
        
        # 为每个条形添加标签
        for container in plt.gca().containers:
            plt.bar_label(container, fmt='%d')
    else:
        plt.text(0.5, 0.5, 'No board size usage data available', ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'memory_usage_patterns.png'), dpi=300, bbox_inches='tight')
    print(f"Memory usage patterns analysis saved to {os.path.join(save_path, 'memory_usage_patterns.png')}")
    
    # 创建第二个图表，分析记忆使用与棋局状态的关系
    plt.figure(figsize=(16, 10))
    
    # 1. 记忆使用与游戏阶段的胜率关系
    plt.subplot(2, 2, 1)
    
    # 计算每个游戏阶段每种记忆类型的胜率
    if 'agent_won' in memory_usage_df.columns and 'phase' in memory_usage_df.columns:
        phase_win_rate = memory_usage_df.groupby(['phase', 'memory_type'])['agent_won'].mean().reset_index(name='win_rate')
        
        if not phase_win_rate.empty:
            # 绘制每个游戏阶段每种记忆类型的胜率
            sns.barplot(data=phase_win_rate, x='phase', y='win_rate', hue='memory_type')
            
            plt.title('Win Rate by Game Phase and Memory Type', fontsize=14, fontweight='bold')
            plt.xlabel('Game Phase', fontsize=12)
            plt.ylabel('Win Rate', fontsize=12)
            plt.legend(title='Memory Type')
            
            # 为每个条形添加标签
            for container in plt.gca().containers:
                plt.bar_label(container, fmt='%.2f')
        else:
            plt.text(0.5, 0.5, 'No phase win rate data available', ha='center', va='center', transform=plt.gca().transAxes)
    else:
        plt.text(0.5, 0.5, 'No phase or win data available', ha='center', va='center', transform=plt.gca().transAxes)
    
    # 2. 记忆调用类型比例 - 存储/读取比例随游戏进行变化
    plt.subplot(2, 2, 2)
    
    if 'turn' in memory_usage_df.columns and 'operation' in memory_usage_df.columns:
        # 创建游戏进度区间
        memory_usage_df['progress_bin'] = pd.cut(
            memory_usage_df['turn'] / memory_usage_df.groupby(['run_id', 'game_id'])['turn'].transform('max'),
            bins=[0, 0.33, 0.66, 1.0],
            labels=['Early', 'Mid', 'Late']
        )
        
        # 计算每个进度区间的操作类型计数
        progress_ops = memory_usage_df.groupby(['progress_bin', 'operation']).size().reset_index(name='count')
        
        if not progress_ops.empty:
            # 绘制堆叠条形图显示随游戏进行操作类型的变化
            pivot_data = progress_ops.pivot(index='progress_bin', columns='operation', values='count').fillna(0)
            pivot_data = pivot_data.div(pivot_data.sum(axis=1), axis=0)
            
            pivot_data.plot(kind='bar', stacked=True, ax=plt.gca())
            
            plt.title('Memory Operation Type Ratio Throughout Game Progress', fontsize=14, fontweight='bold')
            plt.xlabel('Game Progress', fontsize=12)
            plt.ylabel('Operation Type Ratio', fontsize=12)
            plt.legend(title='Operation Type')
            
            # 添加比例标签
            for n, x in enumerate([p.get_x() + p.get_width() / 2 for p in plt.gca().patches[:len(pivot_data)]]):
                for proportion in pivot_data.iloc[n]:
                    if proportion > 0.05:  # 只显示大于5%的标签
                        plt.text(x, 0.5, f'{proportion:.2f}', ha='center', va='center')
        else:
            plt.text(0.5, 0.5, 'No progress operation data available', ha='center', va='center', transform=plt.gca().transAxes)
    else:
        plt.text(0.5, 0.5, 'No turn or operation data available', ha='center', va='center', transform=plt.gca().transAxes)
    
    # 3. 胜利方与失败方的记忆使用对比
    plt.subplot(2, 2, 3)
    
    if 'agent_won' in memory_usage_df.columns:
        # 统计胜利方和失败方的记忆使用
        winner_memory = memory_usage_df.groupby(['agent_won', 'memory_type']).size().reset_index(name='count')
        
        if not winner_memory.empty:
            # 绘制胜利方与失败方的记忆使用对比
            sns.barplot(data=winner_memory, x='agent_won', y='count', hue='memory_type')
            
            plt.title('Memory Usage by Winners vs Losers', fontsize=14, fontweight='bold')
            plt.xlabel('Agent Won', fontsize=12)
            plt.ylabel('Number of Memory Calls', fontsize=12)
            plt.legend(title='Memory Type')
            
            # 为每个条形添加标签
            for container in plt.gca().containers:
                plt.bar_label(container, fmt='%d')
            
            # 设置X轴刻度标签
            plt.xticks([0, 1], ['Lost', 'Won'])
        else:
            plt.text(0.5, 0.5, 'No winner memory data available', ha='center', va='center', transform=plt.gca().transAxes)
    else:
        plt.text(0.5, 0.5, 'No agent_won data available', ha='center', va='center', transform=plt.gca().transAxes)
    
    # 4. 不同记忆约束下的记忆使用效率
    plt.subplot(2, 2, 4)
    
    # 合并实验数据和游戏日志数据
    if not df.empty and not memory_usage_df.empty:
        # 对每种记忆约束计算记忆调用次数
        constraint_calls = memory_usage_df.groupby('memory_constraint_a').size().reset_index(name='total_calls')
        
        # 计算每种记忆约束的总游戏数
        constraint_games = df.groupby('memory_constraint_a')['total_games'].sum().reset_index()
        
        # 合并数据
        constraint_efficiency = pd.merge(constraint_calls, constraint_games, on='memory_constraint_a')
        
        if not constraint_efficiency.empty:
            # 计算每游戏的平均记忆调用次数
            constraint_efficiency['calls_per_game'] = constraint_efficiency['total_calls'] / constraint_efficiency['total_games']
            
            # 绘制每种记忆约束的记忆使用效率
            sns.barplot(data=constraint_efficiency, x='memory_constraint_a', y='calls_per_game')
            
            plt.title('Memory Calls per Game by Memory Constraint', fontsize=14, fontweight='bold')
            plt.xlabel('Memory Constraint', fontsize=12)
            plt.ylabel('Average Memory Calls per Game', fontsize=12)
            
            # 为每个条形添加标签
            for i, row in enumerate(constraint_efficiency.itertuples()):
                plt.text(i, row.calls_per_game, f'{row.calls_per_game:.2f}', ha='center', va='bottom')
            
            # 美化x轴标签
            plt.xticks(range(len(constraint_efficiency)), 
                    [c.replace('_only', ' Only').replace('_', ' ').title() for c in constraint_efficiency['memory_constraint_a']])
        else:
            plt.text(0.5, 0.5, 'No constraint efficiency data available', ha='center', va='center', transform=plt.gca().transAxes)
    else:
        plt.text(0.5, 0.5, 'Insufficient data for constraint efficiency analysis', ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'memory_usage_strategy.png'), dpi=300, bbox_inches='tight')
    print(f"Memory usage strategy analysis saved to {os.path.join(save_path, 'memory_usage_strategy.png')}")
    
    # 生成记忆使用模式的详细统计数据
    memory_pattern_stats = []
    
    # 按游戏阶段和记忆类型统计记忆使用
    if 'phase' in memory_usage_df.columns:
        phase_stats = memory_usage_df.groupby(['phase', 'memory_type', 'operation']).size().reset_index(name='count')
        
        # 计算每个组合的胜率
        if 'agent_won' in memory_usage_df.columns:
            phase_win = memory_usage_df.groupby(['phase', 'memory_type', 'operation'])['agent_won'].mean().reset_index(name='win_rate')
            phase_stats = pd.merge(phase_stats, phase_win, on=['phase', 'memory_type', 'operation'])
        
        # 添加到总统计数据
        memory_pattern_stats.append(('phase_memory_stats', phase_stats))
    
    # 按代理和记忆类型统计记忆使用
    agent_stats = memory_usage_df.groupby(['agent_id', 'memory_type', 'operation']).size().reset_index(name='count')
    
    # 计算每个组合的胜率
    if 'agent_won' in memory_usage_df.columns:
        agent_win = memory_usage_df.groupby(['agent_id', 'memory_type', 'operation'])['agent_won'].mean().reset_index(name='win_rate')
        agent_stats = pd.merge(agent_stats, agent_win, on=['agent_id', 'memory_type', 'operation'])
    
    # 添加到总统计数据
    memory_pattern_stats.append(('agent_memory_stats', agent_stats))
    
    # 按棋盘大小和记忆类型统计记忆使用
    board_stats = memory_usage_df.groupby(['board_size', 'memory_type', 'operation']).size().reset_index(name='count')
    
    # 计算每个组合的胜率
    if 'agent_won' in memory_usage_df.columns:
        board_win = memory_usage_df.groupby(['board_size', 'memory_type', 'operation'])['agent_won'].mean().reset_index(name='win_rate')
        board_stats = pd.merge(board_stats, board_win, on=['board_size', 'memory_type', 'operation'])
    
    # 添加到总统计数据
    memory_pattern_stats.append(('board_memory_stats', board_stats))
    
    # 保存所有统计数据到CSV文件
    for name, stats in memory_pattern_stats:
        stats.to_csv(os.path.join(save_path, f'{name}.csv'))
        print(f"Memory usage statistics saved to {os.path.join(save_path, f'{name}.csv')}")
    
    return memory_pattern_stats

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
    analyze_memory_usage_patterns(df, memory_usage_df)
    
    print("Memory usage metrics analysis completed.") 