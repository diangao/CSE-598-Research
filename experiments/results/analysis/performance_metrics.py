import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .data_loader import load_experiment_data
import logging
import shutil
import json

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_win_rates(df=None, save_path='experiments/results/analysis/figures'):
    """
    Analyze win rates across different memory architectures and board sizes
    
    Args:
        df: Experiment data DataFrame, loaded if None
        save_path: Path to save figures
    """
    if df is None:
        df = load_experiment_data(experiment_type='constrained')
    
    os.makedirs(save_path, exist_ok=True)
    
    # Set a consistent style for all plots
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create win rate analysis charts with improved aesthetics
    plt.figure(figsize=(16, 14))
    
    # Define consistent colors for better visualization
    colors = {'Agent A': '#1f77b4', 'Agent B': '#ff7f0e'}
    
    # 1. Win rates by memory architecture - Enhanced visualization
    plt.subplot(2, 2, 1)
    win_data = []
    
    for _, row in df.iterrows():
        win_data.append({
            'memory_constraint': row['memory_constraint_a'],
            'board_size': row['board_size'],
            'agent': 'Agent A',
            'win_rate': row['agent_a_win_rate'] * 100
        })
        win_data.append({
            'memory_constraint': row['memory_constraint_a'],
            'board_size': row['board_size'],
            'agent': 'Agent B',
            'win_rate': row['agent_b_win_rate'] * 100
        })
    
    win_df = pd.DataFrame(win_data)
    
    # 移除图表中多余的0.0%标签 - 清除当前子图
    plt.clf()
    plt.subplot(2, 2, 1)
    
    # Create more informative barplot
    ax1 = sns.barplot(data=win_df, x='memory_constraint', y='win_rate', hue='agent', palette=colors)
    
    # Add value labels on top of each bar
    for p in ax1.patches:
        height = p.get_height()
        if height > 0.5:  # 只为有意义的高度添加标签
            ax1.text(p.get_x() + p.get_width()/2., height + 1,
                    f'{height:.1f}%',
                    ha="center", fontsize=10)
    
    # Improve labels and title
    plt.title('Win Rates by Memory Architecture and Agent', fontsize=14, fontweight='bold')
    plt.xlabel('Memory Architecture', fontsize=12)
    plt.ylabel('Win Rate (%)', fontsize=12)
    
    # Rename x-axis labels for clarity
    plt.xticks([0, 1], ['Graph Memory Only', 'Vector Memory Only'], fontsize=11)
    
    # Add a horizontal line at 50% to show baseline
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
    
    # Add annotation explaining the implication
    plt.text(0.5, 5, "Hypothesis: Vector memory provides advantage on complex board states",
             ha='center', fontsize=10, style='italic', color='darkblue')
    
    # 2. Win rates by board size - Enhanced visualization
    plt.subplot(2, 2, 2)
    size_data = []
    
    for _, row in df.iterrows():
        size_data.append({
            'memory_constraint': row['memory_constraint_a'],
            'board_size': row['board_size'],
            'agent': 'Agent A',
            'win_rate': row['agent_a_win_rate'] * 100
        })
        size_data.append({
            'memory_constraint': row['memory_constraint_a'],
            'board_size': row['board_size'],
            'agent': 'Agent B',
            'win_rate': row['agent_b_win_rate'] * 100
        })
    
    size_df = pd.DataFrame(size_data)
    
    # 确保按照棋盘大小的正确顺序(从小到大)排序
    board_size_order = sorted(size_df['board_size'].unique())
    
    # 移除图表中多余的0.0%标签 - 清除当前子图
    plt.clf()
    plt.subplot(2, 2, 2)
    
    # Create more informative barplot with correct board size order
    ax2 = sns.barplot(data=size_df, x='board_size', y='win_rate', hue='agent', palette=colors, order=board_size_order)
    
    # Add value labels on top of each bar
    for p in ax2.patches:
        height = p.get_height()
        if height > 0.5:  # 只为有意义的高度添加标签
            ax2.text(p.get_x() + p.get_width()/2., height + 1,
                    f'{height:.1f}%',
                    ha="center", fontsize=10)
    
    # Improve labels and title
    plt.title('Win Rates by Board Size', fontsize=14, fontweight='bold')
    plt.xlabel('Board Size', fontsize=12)
    plt.ylabel('Win Rate (%)', fontsize=12)
    
    # Add a horizontal line at 50% to show balance point
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
    
    # Add annotation explaining the implication
    plt.text(1, 5, "Hypothesis: Board size affects relative agent performance",
             ha='center', fontsize=10, style='italic', color='darkblue')
    
    # 3. Draw rates by memory architecture and board size - Enhanced visualization
    plt.subplot(2, 2, 3)
    draw_data = []
    
    for _, row in df.iterrows():
        draw_data.append({
            'memory_constraint': row['memory_constraint_a'],
            'board_size': row['board_size'],
            'draw_rate': row['draw_rate'] * 100
        })
    
    draw_df = pd.DataFrame(draw_data)
    
    # 确保按照棋盘大小的正确顺序(从小到大)排序
    board_size_order = sorted(draw_df['board_size'].unique())
    
    # 移除图表中多余的0.0%标签 - 清除当前子图
    plt.clf()
    plt.subplot(2, 2, 3)
    
    # Create more informative barplot with better colors and correct board size order
    ax3 = sns.barplot(data=draw_df, x='memory_constraint', y='draw_rate', hue='board_size', palette='viridis', hue_order=board_size_order)
    
    # Add value labels on top of each bar
    for p in ax3.patches:
        height = p.get_height()
        if height > 0.5:  # 只为有意义的高度添加标签
            ax3.text(p.get_x() + p.get_width()/2., height + 0.5,
                    f'{height:.1f}%',
                    ha="center", fontsize=10)
    
    # Improve labels and title
    plt.title('Draw Rates by Memory Architecture and Board Size', fontsize=14, fontweight='bold')
    plt.xlabel('Memory Architecture', fontsize=12)
    plt.ylabel('Draw Rate (%)', fontsize=12)
    
    # Rename x-axis labels for clarity
    plt.xticks([0, 1], ['Graph Memory Only', 'Vector Memory Only'], fontsize=11)
    
    # Add annotation explaining the implication
    plt.text(0.5, 5, "Finding: Draw rates decrease with increasing board complexity",
             ha='center', fontsize=10, style='italic', color='darkblue')
    
    # 4. First-player advantage by memory architecture and board size - Enhanced visualization
    plt.subplot(2, 2, 4)
    advantage_data = []
    
    for _, row in df.iterrows():
        if 'first_player_win_rate' in row:
            advantage_data.append({
                'memory_constraint': row['memory_constraint_a'],
                'board_size': row['board_size'],
                'first_player_advantage': row['first_player_win_rate'] * 100 - 50  # Advantage as deviation from 50%
            })
    
    if advantage_data:
        advantage_df = pd.DataFrame(advantage_data)
        
        # 确保按照棋盘大小的正确顺序(从小到大)排序
        board_size_order = sorted(advantage_df['board_size'].unique())
        
        # 移除图表中多余的0.0%标签 - 清除当前子图
        plt.clf()
        plt.subplot(2, 2, 4)
        
        ax4 = sns.barplot(data=advantage_df, x='memory_constraint', y='first_player_advantage', hue='board_size', palette='viridis', hue_order=board_size_order)
        
        # Add value labels on top of each bar
        for p in ax4.patches:
            height = p.get_height()
            if abs(height) > 1:  # Only add label if there's a significant advantage
                ax4.text(p.get_x() + p.get_width()/2., height + (1 if height > 0 else -3),
                        f'{height:.1f}%',
                        ha="center", fontsize=10)
        
        # Improve labels and title
        plt.title('First-Player Advantage by Memory Architecture and Board Size', fontsize=14, fontweight='bold')
        plt.xlabel('Memory Architecture', fontsize=12)
        plt.ylabel('First-Player Advantage (deviation from 50%)', fontsize=12)
        
        # Rename x-axis labels for clarity
        plt.xticks([0, 1], ['Graph Memory Only', 'Vector Memory Only'], fontsize=11)
        
        # Add a horizontal line at 0 to show baseline (no advantage)
        plt.axhline(y=0, color='red', linestyle='--')
        
        # Add annotation explaining the implication
        plt.text(0.5, -15, "Finding: First-player advantage increases with board size",
                 ha='center', fontsize=10, style='italic', color='darkblue')
    else:
        plt.text(0.5, 0.5, 'No first-player advantage data available', 
                 horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'win_rate_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Win rate analysis plot saved to {os.path.join(save_path, 'win_rate_analysis.png')}")
    
    # Generate win rate statistics table
    win_stats = df.groupby(['memory_constraint_a', 'board_size']).agg({
        'agent_a_win_rate': ['mean', 'std'],
        'agent_b_win_rate': ['mean', 'std'],
        'draw_rate': ['mean', 'std']
    }).reset_index()
    
    # Save statistics to CSV
    win_stats.to_csv(os.path.join(save_path, 'win_rate_statistics.csv'))
    print(f"Win rate statistics saved to {os.path.join(save_path, 'win_rate_statistics.csv')}")
    
    return win_stats

def analyze_token_efficiency(df=None, save_path='experiments/results/analysis/figures'):
    """
    Analyze token usage efficiency across different memory architectures and board sizes
    
    Args:
        df: Experiment data DataFrame, loaded if None
        save_path: Path to save figures
    """
    if df is None:
        df = load_experiment_data(experiment_type='constrained')
    
    os.makedirs(save_path, exist_ok=True)
    
    # Check if token data is available
    token_columns = ['agent_a_total_tokens', 'agent_b_total_tokens']
    if not all(col in df.columns for col in token_columns):
        print("Token usage data not available in dataset")
        return None
    
    # Create token efficiency analysis charts
    plt.figure(figsize=(15, 12))
    
    # 1. Average token usage per game by memory architecture
    plt.subplot(2, 2, 1)
    token_data = []
    
    for _, row in df.iterrows():
        if row['total_games'] > 0:
            token_data.append({
                'memory_constraint': row['memory_constraint_a'],
                'board_size': row['board_size'],
                'agent': 'Agent A',
                'tokens_per_game': row['agent_a_total_tokens'] / row['total_games']
            })
            token_data.append({
                'memory_constraint': row['memory_constraint_a'],
                'board_size': row['board_size'],
                'agent': 'Agent B',
                'tokens_per_game': row['agent_b_total_tokens'] / row['total_games']
            })
    
    if token_data:
        token_df = pd.DataFrame(token_data)
        
        # 移除图表中多余的0.0%标签 - 清除当前子图
        plt.clf()
        plt.subplot(2, 2, 1)
        
        ax = sns.barplot(data=token_df, x='memory_constraint', y='tokens_per_game', hue='agent')
        
        # 为图表添加数值标签
        for p in ax.patches:
            height = p.get_height()
            if height > 100:  # 只为有意义的数值添加标签
                ax.text(p.get_x() + p.get_width()/2., height + 1000,
                        f'{int(height):,}',
                        ha="center", fontsize=9)
        
        plt.title('Average Token Usage per Game by Memory Architecture')
        plt.xlabel('Memory Architecture')
        plt.ylabel('Tokens per Game')
        plt.xticks([0, 1], ['Graph Memory Only', 'Vector Memory Only'], fontsize=11)
    else:
        plt.text(0.5, 0.5, 'No token usage data available', 
                 horizontalalignment='center', verticalalignment='center')
    
    # 2. Average token usage per game by board size
    plt.subplot(2, 2, 2)
    
    if token_data:
        # 确保按照棋盘大小的正确顺序(从小到大)排序
        board_size_order = sorted(token_df['board_size'].unique())
        
        # 移除图表中多余的0.0%标签 - 清除当前子图
        plt.clf()
        plt.subplot(2, 2, 2)
        
        ax = sns.barplot(data=token_df, x='board_size', y='tokens_per_game', hue='agent', order=board_size_order)
        
        # 为图表添加数值标签
        for p in ax.patches:
            height = p.get_height()
            if height > 100:  # 只为有意义的数值添加标签
                ax.text(p.get_x() + p.get_width()/2., height + 1000,
                        f'{int(height):,}',
                        ha="center", fontsize=9)
        
        plt.title('Average Token Usage per Game by Board Size')
        plt.xlabel('Board Size')
        plt.ylabel('Tokens per Game')
    else:
        plt.text(0.5, 0.5, 'No token usage data available', 
                 horizontalalignment='center', verticalalignment='center')
    
    # 3. Win-rate to token ratio by memory architecture
    plt.subplot(2, 2, 3)
    ratio_data = []
    
    for _, row in df.iterrows():
        if row['total_games'] > 0 and row['agent_a_total_tokens'] > 0:
            ratio_data.append({
                'memory_constraint': row['memory_constraint_a'],
                'board_size': row['board_size'],
                'agent': 'Agent A',
                'win_token_ratio': (row['agent_a_win_rate'] * 10000) / (row['agent_a_total_tokens'] / row['total_games'])
            })
        if row['total_games'] > 0 and row['agent_b_total_tokens'] > 0:
            ratio_data.append({
                'memory_constraint': row['memory_constraint_a'],
                'board_size': row['board_size'],
                'agent': 'Agent B',
                'win_token_ratio': (row['agent_b_win_rate'] * 10000) / (row['agent_b_total_tokens'] / row['total_games'])
            })
    
    if ratio_data:
        ratio_df = pd.DataFrame(ratio_data)
        
        # 移除图表中多余的0.0%标签 - 清除当前子图
        plt.clf()
        plt.subplot(2, 2, 3)
        
        ax = sns.barplot(data=ratio_df, x='memory_constraint', y='win_token_ratio', hue='agent')
        
        # 为图表添加数值标签
        for p in ax.patches:
            height = p.get_height()
            if height > 0.01:  # 只为有意义的数值添加标签
                ax.text(p.get_x() + p.get_width()/2., height + 0.02,
                        f'{height:.2f}',
                        ha="center", fontsize=9)
        
        plt.title('Win-Rate to Token Ratio by Memory Architecture (higher is better)')
        plt.xlabel('Memory Architecture')
        plt.ylabel('Win Rate * 10000 / Tokens per Game')
        plt.xticks([0, 1], ['Graph Memory Only', 'Vector Memory Only'], fontsize=11)
    else:
        plt.text(0.5, 0.5, 'No token ratio data available', 
                 horizontalalignment='center', verticalalignment='center')
    
    # 4. Win-rate to token ratio by board size
    plt.subplot(2, 2, 4)
    
    if ratio_data:
        # 确保按照棋盘大小的正确顺序(从小到大)排序
        board_size_order = sorted(ratio_df['board_size'].unique())
        
        # 移除图表中多余的0.0%标签 - 清除当前子图
        plt.clf()
        plt.subplot(2, 2, 4)
        
        ax = sns.barplot(data=ratio_df, x='board_size', y='win_token_ratio', hue='agent', order=board_size_order)
        
        # 为图表添加数值标签
        for p in ax.patches:
            height = p.get_height()
            if height > 0.01:  # 只为有意义的数值添加标签
                ax.text(p.get_x() + p.get_width()/2., height + 0.02,
                        f'{height:.2f}',
                        ha="center", fontsize=9)
        
        plt.title('Win-Rate to Token Ratio by Board Size (higher is better)')
        plt.xlabel('Board Size')
        plt.ylabel('Win Rate * 10000 / Tokens per Game')
    else:
        plt.text(0.5, 0.5, 'No token ratio data available', 
                 horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'token_efficiency_analysis.png'))
    print(f"Token efficiency analysis plot saved to {os.path.join(save_path, 'token_efficiency_analysis.png')}")
    
    # Generate token efficiency statistics table
    token_stats = df.groupby(['memory_constraint_a', 'board_size']).agg({
        'agent_a_total_tokens': ['sum', 'mean'],
        'agent_b_total_tokens': ['sum', 'mean'],
        'total_games': 'sum'
    }).reset_index()
    
    # Calculate derived metrics
    token_stats['agent_a_tokens_per_game'] = token_stats[('agent_a_total_tokens', 'sum')] / token_stats[('total_games', 'sum')]
    token_stats['agent_b_tokens_per_game'] = token_stats[('agent_b_total_tokens', 'sum')] / token_stats[('total_games', 'sum')]
    
    # Save statistics to CSV
    token_stats.to_csv(os.path.join(save_path, 'token_efficiency_statistics.csv'))
    print(f"Token efficiency statistics saved to {os.path.join(save_path, 'token_efficiency_statistics.csv')}")
    
    return token_stats

def analyze_performance_retention(df=None, save_path='experiments/results/analysis/figures'):
    """
    Analyze how well agents maintain performance across increasing board sizes
    
    Args:
        df: Experiment data DataFrame, loaded if None
        save_path: Path to save figures
    """
    if df is None:
        df = load_experiment_data(experiment_type='constrained')
    
    os.makedirs(save_path, exist_ok=True)
    
    # Create performance retention charts
    plt.figure(figsize=(15, 10))
    
    # Baseline performance at smallest board size
    smallest_size = df['board_size'].min()
    baseline = df[df['board_size'] == smallest_size].groupby('memory_constraint_a').agg({
        'agent_a_win_rate': 'mean',
        'agent_b_win_rate': 'mean'
    })
    
    # 1. Performance retention by board size and memory architecture (Agent A)
    plt.subplot(2, 2, 1)
    retention_data_a = []
    
    for _, row in df.iterrows():
        base_a = baseline.loc[row['memory_constraint_a'], 'agent_a_win_rate']
        retention_data_a.append({
            'memory_constraint': row['memory_constraint_a'],
            'board_size': row['board_size'],
            'retention_rate': (row['agent_a_win_rate'] / base_a if base_a > 0 else 0) * 100
        })
    
    retention_df_a = pd.DataFrame(retention_data_a)
    
    # 确保按照棋盘大小的正确顺序(从小到大)排序
    board_size_order = sorted(retention_df_a['board_size'].unique())
    
    for constraint in sorted(retention_df_a['memory_constraint'].unique()):
        data = retention_df_a[retention_df_a['memory_constraint'] == constraint]
        data = data.sort_values(by='board_size')  # 确保数据按棋盘大小排序
        plt.plot(data['board_size'], data['retention_rate'], marker='o', label=f"{constraint}")
    
    plt.axhline(y=100, color='gray', linestyle='--')
    plt.title('Agent A: Performance Retention Rate by Board Size')
    plt.xlabel('Board Size')
    plt.ylabel('Performance Retention (% of 3x3 baseline)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(labels=['graph_only', 'vector_only'])
    plt.xticks(board_size_order)  # 确保X轴刻度按正确顺序显示
    
    # 2. Performance retention by board size and memory architecture (Agent B)
    plt.subplot(2, 2, 2)
    retention_data_b = []
    
    for _, row in df.iterrows():
        base_b = baseline.loc[row['memory_constraint_a'], 'agent_b_win_rate']
        retention_data_b.append({
            'memory_constraint': row['memory_constraint_a'],
            'board_size': row['board_size'],
            'retention_rate': (row['agent_b_win_rate'] / base_b if base_b > 0 else 0) * 100
        })
    
    retention_df_b = pd.DataFrame(retention_data_b)
    
    # 确保按照棋盘大小的正确顺序(从小到大)排序
    board_size_order = sorted(retention_df_b['board_size'].unique())
    
    for constraint in sorted(retention_df_b['memory_constraint'].unique()):
        data = retention_df_b[retention_df_b['memory_constraint'] == constraint]
        data = data.sort_values(by='board_size')  # 确保数据按棋盘大小排序
        plt.plot(data['board_size'], data['retention_rate'], marker='o', label=f"{constraint}")
    
    plt.axhline(y=100, color='gray', linestyle='--')
    plt.title('Agent B: Performance Retention Rate by Board Size')
    plt.xlabel('Board Size')
    plt.ylabel('Performance Retention (% of 3x3 baseline)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(labels=['graph_only', 'vector_only'])
    plt.xticks(board_size_order)  # 确保X轴刻度按正确顺序显示
    
    # 3. Comparative retention performance (Agent A vs B)
    plt.subplot(2, 2, 3)
    combined_data = []
    
    for _, row in df.iterrows():
        if row['board_size'] != smallest_size:  # Skip baseline
            base_a = baseline.loc[row['memory_constraint_a'], 'agent_a_win_rate']
            base_b = baseline.loc[row['memory_constraint_a'], 'agent_b_win_rate']
            
            combined_data.append({
                'memory_constraint': row['memory_constraint_a'],
                'board_size': row['board_size'],
                'agent': 'Agent A',
                'retention_rate': (row['agent_a_win_rate'] / base_a if base_a > 0 else 0) * 100
            })
            combined_data.append({
                'memory_constraint': row['memory_constraint_a'],
                'board_size': row['board_size'],
                'agent': 'Agent B',
                'retention_rate': (row['agent_b_win_rate'] / base_b if base_b > 0 else 0) * 100
            })
    
    if combined_data:
        combined_df = pd.DataFrame(combined_data)
        # 确保按照棋盘大小的正确顺序(从小到大)排序
        board_size_order = sorted(combined_df['board_size'].unique())
        sns.barplot(data=combined_df, x='board_size', y='retention_rate', hue='agent', order=board_size_order)
        plt.axhline(y=100, color='gray', linestyle='--')
        plt.title('Performance Retention Comparison between Agents')
        plt.xlabel('Board Size')
        plt.ylabel('Performance Retention (% of 3x3 baseline)')
    else:
        plt.text(0.5, 0.5, 'No comparative retention data available',
                horizontalalignment='center', verticalalignment='center')
    
    # 4. Memory architecture performance retention comparison
    plt.subplot(2, 2, 4)
    arch_comparison = []
    
    for _, row in df.iterrows():
        if row['board_size'] != smallest_size:  # Skip baseline
            base_a = baseline.loc[row['memory_constraint_a'], 'agent_a_win_rate']
            base_b = baseline.loc[row['memory_constraint_a'], 'agent_b_win_rate']
            
            # Average retention across both agents
            avg_retention = (
                (row['agent_a_win_rate'] / base_a if base_a > 0 else 0) +
                (row['agent_b_win_rate'] / base_b if base_b > 0 else 0)
            ) * 50  # Average and convert to percentage
            
            arch_comparison.append({
                'memory_constraint': row['memory_constraint_a'],
                'board_size': row['board_size'],
                'avg_retention': avg_retention
            })
    
    if arch_comparison:
        arch_df = pd.DataFrame(arch_comparison)
        # 确保按照棋盘大小的正确顺序(从小到大)排序
        board_size_order = sorted(arch_df['board_size'].unique())
        sns.barplot(data=arch_df, x='memory_constraint', y='avg_retention', hue='board_size', hue_order=board_size_order)
        plt.axhline(y=100, color='gray', linestyle='--')
        plt.title('Performance Retention by Memory Architecture')
        plt.xlabel('Memory Architecture')
        plt.ylabel('Average Performance Retention (%)')
    else:
        plt.text(0.5, 0.5, 'No architecture comparison data available',
                horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'performance_retention_analysis.png'))
    print(f"Performance retention analysis plot saved to {os.path.join(save_path, 'performance_retention_analysis.png')}")
    
    # Generate performance retention statistics
    a_retention_pivot = retention_df_a.pivot(index='memory_constraint', columns='board_size', values='retention_rate')
    b_retention_pivot = retention_df_b.pivot(index='memory_constraint', columns='board_size', values='retention_rate')
    
    # Combine the two pivoted dataframes
    retention_stats = pd.DataFrame()
    for col in a_retention_pivot.columns:
        retention_stats[f'agent_a_retention_{col}'] = a_retention_pivot[col]
    for col in b_retention_pivot.columns:
        retention_stats[f'agent_b_retention_{col}'] = b_retention_pivot[col]
    
    # Save statistics to CSV
    retention_stats.to_csv(os.path.join(save_path, 'performance_retention_statistics.csv'))
    print(f"Performance retention statistics saved to {os.path.join(save_path, 'performance_retention_statistics.csv')}")
    
    return retention_stats

def analyze_memory_calls(df=None, save_path='experiments/results/analysis/figures'):
    """
    Analyze memory call patterns across different board sizes and memory architectures
    
    Args:
        df: Experiment data DataFrame, loaded if None
        save_path: Path to save figures
    """
    if df is None:
        df = load_experiment_data(experiment_type='constrained')
    
    os.makedirs(save_path, exist_ok=True)
    
    # 这里不需要两个figure，去掉第一个
    # plt.figure(figsize=(10, 8))
    # plt.suptitle('Memory Calls by Board Size and Memory Architecture', fontsize=16, fontweight='bold')
    
    # Prepare data for memory calls chart
    memory_data = []
    
    for _, row in df.iterrows():
        # Agent A memory calls
        if 'agent_a_graph_calls' in row:
            memory_data.append({
                'board_size': row['board_size'],
                'agent': 'Agent A',
                'memory_type': 'graph_only',
                'calls': row['agent_a_graph_calls'] if row['memory_constraint_a'] == 'graph_only' else 0
            })
        if 'agent_a_vector_calls' in row:
            memory_data.append({
                'board_size': row['board_size'],
                'agent': 'Agent A',
                'memory_type': 'vector_only',
                'calls': row['agent_a_vector_calls'] if row['memory_constraint_a'] == 'vector_only' else 0
            })
            
        # Agent B memory calls
        if 'agent_b_graph_calls' in row:
            memory_data.append({
                'board_size': row['board_size'],
                'agent': 'Agent B',
                'memory_type': 'graph_only',
                'calls': row['agent_b_graph_calls'] if row['memory_constraint_a'] == 'graph_only' else 0
            })
        if 'agent_b_vector_calls' in row:
            memory_data.append({
                'board_size': row['board_size'],
                'agent': 'Agent B',
                'memory_type': 'vector_only',
                'calls': row['agent_b_vector_calls'] if row['memory_constraint_a'] == 'vector_only' else 0
            })
    
    # Create DataFrame
    memory_df = pd.DataFrame(memory_data)
    
    # Filter out zero values
    memory_df = memory_df[memory_df['calls'] > 0]
    
    # Group by agent, board_size, and memory_type to get average calls
    memory_summary = memory_df.groupby(['agent', 'board_size', 'memory_type'])['calls'].mean().reset_index()
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # 确保按照棋盘大小的正确顺序(从小到大)排序
    board_size_order = sorted(memory_summary['board_size'].unique())
    
    # 清除图表和轴标签中任何可能的0.0%
    plt.clf()
    
    # Plot lines with different colors for each agent and memory type combination
    colors = {'Agent A': {'graph_only': '#1f77b4', 'vector_only': '#ff7f0e'}, 
              'Agent B': {'graph_only': '#2ca02c', 'vector_only': '#d62728'}}
    
    for agent in memory_summary['agent'].unique():
        for memory_type in memory_summary['memory_type'].unique():
            data = memory_summary[(memory_summary['agent'] == agent) & (memory_summary['memory_type'] == memory_type)]
            data = data.sort_values(by='board_size')  # Ensure data is sorted by board size
            if not data.empty:
                plt.plot(data['board_size'], data['calls'], marker='o', 
                         label=f"{agent} ({memory_type})",
                         linewidth=2,
                         color=colors[agent][memory_type])
    
    # 添加数据标签
    for agent in memory_summary['agent'].unique():
        for memory_type in memory_summary['memory_type'].unique():
            data = memory_summary[(memory_summary['agent'] == agent) & (memory_summary['memory_type'] == memory_type)]
            data = data.sort_values(by='board_size')  # Ensure data is sorted by board size
            if not data.empty:
                for i, row in data.iterrows():
                    plt.text(row['board_size'], row['calls'] + 0.5, 
                             f"{int(row['calls'])}", 
                             ha='center', 
                             fontsize=9)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Memory Calls by Board Size and Memory Architecture', fontsize=14, fontweight='bold')
    plt.xlabel('Board Size', fontsize=12)
    plt.ylabel('Average Memory Calls', fontsize=12)
    plt.xticks(board_size_order)  # 确保X轴刻度按正确顺序显示
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, 'memory_calls_by_board_size.png'), dpi=300, bbox_inches='tight')
    print(f"Memory calls analysis plot saved to {os.path.join(save_path, 'memory_calls_by_board_size.png')}")
    
    # Generate memory call statistics
    memory_stats = memory_summary.pivot_table(index=['agent', 'memory_type'], 
                                             columns='board_size', 
                                             values='calls').reset_index()
    
    # Save statistics to CSV
    memory_stats.to_csv(os.path.join(save_path, 'memory_call_statistics.csv'))
    print(f"Memory call statistics saved to {os.path.join(save_path, 'memory_call_statistics.csv')}")
    
    return memory_stats

def analyze_memory_baseline_comparison(df=None, save_path='experiments/results/analysis/figures/memory_baseline'):
    """
    Analyze memory comparison baseline experiments
    
    Args:
        df: Experiment data DataFrame, loaded if None
        save_path: Path to save figures
    """
    if df is None:
        # 明确指定experiment_type为'memory_comparison'，确保从memory_baseline目录读取数据
        df = load_experiment_data(experiment_type='memory_comparison')
    
    os.makedirs(save_path, exist_ok=True)
    
    # Set a consistent style for all plots
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # If no data is found
    if df.empty:
        print("No memory comparison data found")
        return None
    
    # 从文件名中提取agent类型信息
    df['memory_agent_type'] = df['filename'].str.extract(r'compare_agent_([ab])_')
    
    # 添加一个新列来保存实际的记忆类型，考虑到Agent B实验中的列关系
    df['actual_memory_type'] = 'unknown'
    
    # 对于Agent A：记忆类型在memory_constraint_a中
    # 对于Agent B：记忆类型在memory_constraint_b中
    for idx, row in df.iterrows():
        if row['memory_agent_type'] == 'a':
            df.at[idx, 'actual_memory_type'] = row['memory_constraint_a']
        else:  # agent_type == 'b'
            df.at[idx, 'actual_memory_type'] = row['memory_constraint_b']
    
    # 创建完整比较图表，包括胜、负、平局情况
    plt.figure(figsize=(18, 16))
    
    # 定义颜色
    colors = {'Win': '#1f77b4', 'Draw': '#ff7f0e', 'Loss': '#d62728'}
    
    # 1. 完整结果比较 - Agent A with Graph Memory
    plt.subplot(2, 2, 1)
    agent_a_graph = df[(df['memory_agent_type'] == 'a') & (df['actual_memory_type'] == 'graph_only')]
    if not agent_a_graph.empty:
        # 聚合多个实验的结果
        if len(agent_a_graph) > 1:
            # 多行结果需要聚合
            a_graph_results = agent_a_graph.agg({
                'agent_a_win_rate': 'mean',
                'agent_b_win_rate': 'mean',
                'draw_rate': 'mean',
                'total_games': 'sum'
            })
        else:
            # 单行结果直接使用第一行
            a_graph_results = agent_a_graph.iloc[0]
        
        # 准备数据
        results_data = []
        
        # 胜率 - 有记忆
        results_data.append({
            'result_type': 'Win',
            'agent_type': 'With Memory',
            'percentage': a_graph_results['agent_a_win_rate'] * 100
        })
        
        # 平局率 - 有记忆
        results_data.append({
            'result_type': 'Draw',
            'agent_type': 'With Memory',
            'percentage': a_graph_results['draw_rate'] * 100
        })
        
        # 负率 - 有记忆
        loss_rate = 1 - a_graph_results['agent_a_win_rate'] - a_graph_results['draw_rate']
        results_data.append({
            'result_type': 'Loss',
            'agent_type': 'With Memory',
            'percentage': loss_rate * 100
        })
        
        # 胜率 - 无记忆
        results_data.append({
            'result_type': 'Win',
            'agent_type': 'No Memory',
            'percentage': a_graph_results['agent_b_win_rate'] * 100
        })
        
        # 平局率 - 无记忆 (平局率相同)
        results_data.append({
            'result_type': 'Draw',
            'agent_type': 'No Memory',
            'percentage': a_graph_results['draw_rate'] * 100
        })
        
        # 负率 - 无记忆
        loss_rate_opponent = 1 - a_graph_results['agent_b_win_rate'] - a_graph_results['draw_rate']
        results_data.append({
            'result_type': 'Loss',
            'agent_type': 'No Memory',
            'percentage': loss_rate_opponent * 100
        })
        
        results_df = pd.DataFrame(results_data)
        
        # 确保每个组合是唯一的
        results_df = results_df.drop_duplicates(['agent_type', 'result_type'])
        
        # 使用堆叠条形图展示完整结果
        ax1 = plt.gca()
        
        # 对于每个代理类型，分别绘制堆叠条形图
        for i, agent_type in enumerate(['With Memory', 'No Memory']):
            agent_data = results_df[results_df['agent_type'] == agent_type]
            bottom = 0
            for result_type in ['Win', 'Draw', 'Loss']:
                result_row = agent_data[agent_data['result_type'] == result_type]
                if not result_row.empty:
                    height = result_row['percentage'].values[0]
                    bar = ax1.bar(i, height, bottom=bottom, color=colors[result_type], label=result_type if i == 0 else "")
                    # 为所有值添加标签，不论大小
                    ax1.text(i, bottom + height/2, f'{height:.1f}%', ha='center', va='center', fontweight='bold')
                    bottom += height
        
        plt.title('Agent A with Graph Memory: Complete Results', fontsize=14, fontweight='bold')
        plt.xticks([0, 1], ['With Memory', 'No Memory'])
        plt.xlabel('Agent Type', fontsize=12)
        plt.ylabel('Percentage (%)', fontsize=12)
        plt.ylim(0, 100)
        
        # 创建自定义图例
        handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in ['Win', 'Draw', 'Loss']]
        plt.legend(handles, ['Win', 'Draw', 'Loss'], title='Result')
    
    # 2. 完整结果比较 - Agent A with Vector Memory
    plt.subplot(2, 2, 2)
    agent_a_vector = df[(df['memory_agent_type'] == 'a') & (df['actual_memory_type'] == 'vector_only')]
    if not agent_a_vector.empty:
        # 聚合多个实验的结果
        if len(agent_a_vector) > 1:
            # 多行结果需要聚合
            a_vector_results = agent_a_vector.agg({
                'agent_a_win_rate': 'mean',
                'agent_b_win_rate': 'mean',
                'draw_rate': 'mean',
                'total_games': 'sum'
            })
        else:
            # 单行结果直接使用第一行
            a_vector_results = agent_a_vector.iloc[0]
            
        # 准备数据
        results_data = []
        
        # 胜率 - 有记忆
        results_data.append({
            'result_type': 'Win',
            'agent_type': 'With Memory',
            'percentage': a_vector_results['agent_a_win_rate'] * 100
        })
        
        # 平局率 - 有记忆
        results_data.append({
            'result_type': 'Draw',
            'agent_type': 'With Memory',
            'percentage': a_vector_results['draw_rate'] * 100
        })
        
        # 负率 - 有记忆
        loss_rate = 1 - a_vector_results['agent_a_win_rate'] - a_vector_results['draw_rate']
        results_data.append({
            'result_type': 'Loss',
            'agent_type': 'With Memory',
            'percentage': loss_rate * 100
        })
        
        # 胜率 - 无记忆
        results_data.append({
            'result_type': 'Win',
            'agent_type': 'No Memory',
            'percentage': a_vector_results['agent_b_win_rate'] * 100
        })
        
        # 平局率 - 无记忆 (平局率相同)
        results_data.append({
            'result_type': 'Draw',
            'agent_type': 'No Memory',
            'percentage': a_vector_results['draw_rate'] * 100
        })
        
        # 负率 - 无记忆
        loss_rate_opponent = 1 - a_vector_results['agent_b_win_rate'] - a_vector_results['draw_rate']
        results_data.append({
            'result_type': 'Loss',
            'agent_type': 'No Memory',
            'percentage': loss_rate_opponent * 100
        })
        
        results_df = pd.DataFrame(results_data)
        
        # 确保每个组合是唯一的
        results_df = results_df.drop_duplicates(['agent_type', 'result_type'])
        
        # 使用堆叠条形图展示完整结果
        ax2 = plt.gca()
        
        # 对于每个代理类型，分别绘制堆叠条形图
        for i, agent_type in enumerate(['With Memory', 'No Memory']):
            agent_data = results_df[results_df['agent_type'] == agent_type]
            bottom = 0
            for result_type in ['Win', 'Draw', 'Loss']:
                result_row = agent_data[agent_data['result_type'] == result_type]
                if not result_row.empty:
                    height = result_row['percentage'].values[0]
                    bar = ax2.bar(i, height, bottom=bottom, color=colors[result_type], label=result_type if i == 0 else "")
                    # 为所有值添加标签，不论大小
                    ax2.text(i, bottom + height/2, f'{height:.1f}%', ha='center', va='center', fontweight='bold')
                    bottom += height
        
        plt.title('Agent A with Vector Memory: Complete Results', fontsize=14, fontweight='bold')
        plt.xticks([0, 1], ['With Memory', 'No Memory'])
        plt.xlabel('Agent Type', fontsize=12)
        plt.ylabel('Percentage (%)', fontsize=12)
        plt.ylim(0, 100)
        
        # 创建自定义图例
        handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in ['Win', 'Draw', 'Loss']]
        plt.legend(handles, ['Win', 'Draw', 'Loss'], title='Result')
    
    # 3. 完整结果比较 - Agent B with Graph Memory
    plt.subplot(2, 2, 3)
    agent_b_graph = df[(df['memory_agent_type'] == 'b') & (df['actual_memory_type'] == 'graph_only')]
    if not agent_b_graph.empty:
        # 聚合多个实验的结果
        if len(agent_b_graph) > 1:
            # 多行结果需要聚合
            b_graph_results = agent_b_graph.agg({
                'agent_a_win_rate': 'mean',
                'agent_b_win_rate': 'mean',
                'draw_rate': 'mean',
                'total_games': 'sum'
            })
        else:
            # 单行结果直接使用第一行
            b_graph_results = agent_b_graph.iloc[0]
        
        # 准备数据
        results_data = []
        
        # 胜率 - 有记忆 (Agent B)
        results_data.append({
            'result_type': 'Win',
            'agent_type': 'With Memory',
            'percentage': b_graph_results['agent_b_win_rate'] * 100
        })
        
        # 平局率 - 有记忆
        results_data.append({
            'result_type': 'Draw',
            'agent_type': 'With Memory',
            'percentage': b_graph_results['draw_rate'] * 100
        })
        
        # 负率 - 有记忆
        loss_rate = 1 - b_graph_results['agent_b_win_rate'] - b_graph_results['draw_rate']
        results_data.append({
            'result_type': 'Loss',
            'agent_type': 'With Memory',
            'percentage': loss_rate * 100
        })
        
        # 胜率 - 无记忆 (Agent A)
        results_data.append({
            'result_type': 'Win',
            'agent_type': 'No Memory',
            'percentage': b_graph_results['agent_a_win_rate'] * 100
        })
        
        # 平局率 - 无记忆 (平局率相同)
        results_data.append({
            'result_type': 'Draw',
            'agent_type': 'No Memory',
            'percentage': b_graph_results['draw_rate'] * 100
        })
        
        # 负率 - 无记忆
        loss_rate_opponent = 1 - b_graph_results['agent_a_win_rate'] - b_graph_results['draw_rate']
        results_data.append({
            'result_type': 'Loss',
            'agent_type': 'No Memory',
            'percentage': loss_rate_opponent * 100
        })
        
        results_df = pd.DataFrame(results_data)
        
        # 确保每个组合是唯一的
        results_df = results_df.drop_duplicates(['agent_type', 'result_type'])
        
        # 使用堆叠条形图展示完整结果
        ax3 = plt.gca()
        
        # 对于每个代理类型，分别绘制堆叠条形图
        for i, agent_type in enumerate(['With Memory', 'No Memory']):
            agent_data = results_df[results_df['agent_type'] == agent_type]
            bottom = 0
            for result_type in ['Win', 'Draw', 'Loss']:
                result_row = agent_data[agent_data['result_type'] == result_type]
                if not result_row.empty:
                    height = result_row['percentage'].values[0]
                    bar = ax3.bar(i, height, bottom=bottom, color=colors[result_type], label=result_type if i == 0 else "")
                    # 为所有值添加标签，不论大小
                    ax3.text(i, bottom + height/2, f'{height:.1f}%', ha='center', va='center', fontweight='bold')
                    bottom += height
        
        plt.title('Agent B with Graph Memory: Complete Results', fontsize=14, fontweight='bold')
        plt.xticks([0, 1], ['With Memory', 'No Memory'])
        plt.xlabel('Agent Type', fontsize=12)
        plt.ylabel('Percentage (%)', fontsize=12)
        plt.ylim(0, 100)
        
        # 创建自定义图例
        handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in ['Win', 'Draw', 'Loss']]
        plt.legend(handles, ['Win', 'Draw', 'Loss'], title='Result')
    
    # 4. 完整结果比较 - Agent B with Vector Memory
    plt.subplot(2, 2, 4)
    agent_b_vector = df[(df['memory_agent_type'] == 'b') & (df['actual_memory_type'] == 'vector_only')]
    if not agent_b_vector.empty:
        # 聚合多个实验的结果
        if len(agent_b_vector) > 1:
            # 多行结果需要聚合
            b_vector_results = agent_b_vector.agg({
                'agent_a_win_rate': 'mean',
                'agent_b_win_rate': 'mean',
                'draw_rate': 'mean',
                'total_games': 'sum'
            })
        else:
            # 单行结果直接使用第一行
            b_vector_results = agent_b_vector.iloc[0]
        
        # 准备数据
        results_data = []
        
        # 胜率 - 有记忆 (Agent B)
        results_data.append({
            'result_type': 'Win',
            'agent_type': 'With Memory',
            'percentage': b_vector_results['agent_b_win_rate'] * 100
        })
        
        # 平局率 - 有记忆
        results_data.append({
            'result_type': 'Draw',
            'agent_type': 'With Memory',
            'percentage': b_vector_results['draw_rate'] * 100
        })
        
        # 负率 - 有记忆
        loss_rate = 1 - b_vector_results['agent_b_win_rate'] - b_vector_results['draw_rate']
        results_data.append({
            'result_type': 'Loss',
            'agent_type': 'With Memory',
            'percentage': loss_rate * 100
        })
        
        # 胜率 - 无记忆 (Agent A)
        results_data.append({
            'result_type': 'Win',
            'agent_type': 'No Memory',
            'percentage': b_vector_results['agent_a_win_rate'] * 100
        })
        
        # 平局率 - 无记忆 (平局率相同)
        results_data.append({
            'result_type': 'Draw',
            'agent_type': 'No Memory',
            'percentage': b_vector_results['draw_rate'] * 100
        })
        
        # 负率 - 无记忆
        loss_rate_opponent = 1 - b_vector_results['agent_a_win_rate'] - b_vector_results['draw_rate']
        results_data.append({
            'result_type': 'Loss',
            'agent_type': 'No Memory',
            'percentage': loss_rate_opponent * 100
        })
        
        results_df = pd.DataFrame(results_data)
        
        # 确保每个组合是唯一的
        results_df = results_df.drop_duplicates(['agent_type', 'result_type'])
        
        # 使用堆叠条形图展示完整结果
        ax4 = plt.gca()
        
        # 对于每个代理类型，分别绘制堆叠条形图
        for i, agent_type in enumerate(['With Memory', 'No Memory']):
            agent_data = results_df[results_df['agent_type'] == agent_type]
            bottom = 0
            for result_type in ['Win', 'Draw', 'Loss']:
                result_row = agent_data[agent_data['result_type'] == result_type]
                if not result_row.empty:
                    height = result_row['percentage'].values[0]
                    bar = ax4.bar(i, height, bottom=bottom, color=colors[result_type], label=result_type if i == 0 else "")
                    # 为所有值添加标签，不论大小
                    ax4.text(i, bottom + height/2, f'{height:.1f}%', ha='center', va='center', fontweight='bold')
                    bottom += height
        
        plt.title('Agent B with Vector Memory: Complete Results', fontsize=14, fontweight='bold')
        plt.xticks([0, 1], ['With Memory', 'No Memory'])
        plt.xlabel('Agent Type', fontsize=12)
        plt.ylabel('Percentage (%)', fontsize=12)
        plt.ylim(0, 100)
        
        # 创建自定义图例
        handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in ['Win', 'Draw', 'Loss']]
        plt.legend(handles, ['Win', 'Draw', 'Loss'], title='Result')
    
    # 保存完整结果比较图
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'memory_complete_results.png'), dpi=300, bbox_inches='tight')
    print(f"Complete memory results analysis saved to {os.path.join(save_path, 'memory_complete_results.png')}")
    
    # 保留原始的胜率对比图 - 在新的Figure中
    plt.figure(figsize=(16, 14))
    
    # Define consistent colors for better visualization
    colors = {'With Memory': '#1f77b4', 'No Memory': '#ff7f0e'}
    
    # 1. Win rates by memory architecture for Agent A
    plt.subplot(2, 2, 1)
    agent_a_data = df[df['memory_agent_type'] == 'a']
    if not agent_a_data.empty:
        win_data = []
        for _, row in agent_a_data.iterrows():
            win_data.append({
                'memory_type': row['actual_memory_type'],
                'board_size': row['board_size'],
                'agent_type': 'With Memory',
                'win_rate': row['agent_a_win_rate'] * 100
            })
            win_data.append({
                'memory_type': 'baseline',
                'board_size': row['board_size'],
                'agent_type': 'No Memory',
                'win_rate': row['agent_b_win_rate'] * 100
            })
        
        win_df = pd.DataFrame(win_data)
        
        # 绘制条形图
        sns.barplot(data=win_df[win_df['memory_type'] != 'baseline'], 
                   x='memory_type', y='win_rate', 
                   hue='agent_type', palette=colors, ax=plt.gca())
        
        # Add value labels
        for p in plt.gca().patches:
            height = p.get_height()
            if height > 0.5:
                plt.gca().text(p.get_x() + p.get_width()/2., height + 1,
                        f'{height:.1f}%',
                        ha="center", fontsize=10)
        
        plt.title('Agent A: Memory vs No Memory Win Rates', fontsize=14, fontweight='bold')
        plt.xlabel('Memory Type', fontsize=12)
        plt.ylabel('Win Rate (%)', fontsize=12)
        
        # 简化标签设置，不过滤任何值
        plt.xticks(rotation=0)
        memory_types = sorted([t for t in win_df['memory_type'].unique() if t != 'baseline'])
        plt.gca().set_xticklabels([t.replace('_only', ' Memory').title() for t in memory_types])
        
        plt.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
    
    # 2. Win rates by memory architecture for Agent B
    plt.subplot(2, 2, 2)
    agent_b_data = df[df['memory_agent_type'] == 'b']
    if not agent_b_data.empty:
        win_data = []
        for _, row in agent_b_data.iterrows():
            win_data.append({
                'memory_type': row['actual_memory_type'],
                'board_size': row['board_size'],
                'agent_type': 'With Memory',
                'win_rate': row['agent_b_win_rate'] * 100
            })
            win_data.append({
                'memory_type': 'baseline',
                'board_size': row['board_size'],
                'agent_type': 'No Memory',
                'win_rate': row['agent_a_win_rate'] * 100
            })
        
        win_df = pd.DataFrame(win_data)
        
        # 绘制条形图
        sns.barplot(data=win_df[win_df['memory_type'] != 'baseline'], 
                   x='memory_type', y='win_rate', 
                   hue='agent_type', palette=colors, ax=plt.gca())
        
        # Add value labels
        for p in plt.gca().patches:
            height = p.get_height()
            if height > 0.5:
                plt.gca().text(p.get_x() + p.get_width()/2., height + 1,
                        f'{height:.1f}%',
                        ha="center", fontsize=10)
        
        plt.title('Agent B: Memory vs No Memory Win Rates', fontsize=14, fontweight='bold')
        plt.xlabel('Memory Type', fontsize=12)
        plt.ylabel('Win Rate (%)', fontsize=12)
        
        # 简化标签设置，不过滤任何值
        plt.xticks(rotation=0)
        memory_types = sorted([t for t in win_df['memory_type'].unique() if t != 'baseline'])
        plt.gca().set_xticklabels([t.replace('_only', ' Memory').title() for t in memory_types])
        
        plt.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
    
    # 3. Board size comparison for Agent A
    plt.subplot(2, 2, 3)
    if not agent_a_data.empty:
        board_data = []
        for _, row in agent_a_data.iterrows():
            board_data.append({
                'memory_type': row['actual_memory_type'],
                'board_size': row['board_size'],
                'agent_type': 'With Memory',
                'win_rate': row['agent_a_win_rate'] * 100
            })
            board_data.append({
                'memory_type': 'baseline',
                'board_size': row['board_size'],
                'agent_type': 'No Memory',
                'win_rate': row['agent_b_win_rate'] * 100
            })
        
        board_df = pd.DataFrame(board_data)
        
        # 确保棋盘大小按照从小到大排序
        board_sizes = sorted(board_df['board_size'].unique(), 
                            key=lambda x: int(str(x).split('x')[0] if 'x' in str(x) else x))
        
        ax3 = sns.barplot(data=board_df, x='board_size', y='win_rate', hue='agent_type', 
                         palette=colors, order=board_sizes)
        
        # Add value labels
        for p in ax3.patches:
            height = p.get_height()
            if height > 0.5:
                ax3.text(p.get_x() + p.get_width()/2., height + 1,
                        f'{height:.1f}%',
                        ha="center", fontsize=10)
        
        plt.title('Agent A: Memory Impact by Board Size', fontsize=14, fontweight='bold')
        plt.xlabel('Board Size', fontsize=12)
        plt.ylabel('Win Rate (%)', fontsize=12)
        plt.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
    
    # 4. Board size comparison for Agent B
    plt.subplot(2, 2, 4)
    if not agent_b_data.empty:
        board_data = []
        for _, row in agent_b_data.iterrows():
            board_data.append({
                'memory_type': row['actual_memory_type'],
                'board_size': row['board_size'],
                'agent_type': 'With Memory',
                'win_rate': row['agent_b_win_rate'] * 100
            })
            board_data.append({
                'memory_type': 'baseline',
                'board_size': row['board_size'],
                'agent_type': 'No Memory',
                'win_rate': row['agent_a_win_rate'] * 100
            })
        
        board_df = pd.DataFrame(board_data)
        
        # 确保棋盘大小按照从小到大排序
        board_sizes = sorted(board_df['board_size'].unique(), 
                            key=lambda x: int(str(x).split('x')[0] if 'x' in str(x) else x))
        
        ax4 = sns.barplot(data=board_df, x='board_size', y='win_rate', hue='agent_type', 
                         palette=colors, order=board_sizes)
        
        # Add value labels
        for p in ax4.patches:
            height = p.get_height()
            if height > 0.5:
                ax4.text(p.get_x() + p.get_width()/2., height + 1,
                        f'{height:.1f}%',
                        ha="center", fontsize=10)
        
        plt.title('Agent B: Memory Impact by Board Size', fontsize=14, fontweight='bold')
        plt.xlabel('Board Size', fontsize=12)
        plt.ylabel('Win Rate (%)', fontsize=12)
        plt.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'memory_baseline_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Memory baseline comparison analysis saved to {os.path.join(save_path, 'memory_baseline_comparison.png')}")
    
    return df

def analyze_memory_token_efficiency(df=None, save_path='experiments/results/analysis/figures/memory_baseline'):
    """
    Analyze token usage efficiency for memory vs. no-memory agents
    
    Args:
        df: Experiment data DataFrame, loaded if None
        save_path: Path to save figures
    """
    if df is None:
        # 专门加载memory_comparison类型的实验数据
        df = load_experiment_data(experiment_type='memory_comparison')
    
    os.makedirs(save_path, exist_ok=True)
    
    # 检查数据是否可用
    if df.empty:
        print("No memory comparison data found")
        return None
    
    # 从文件名中提取agent类型信息
    if 'memory_agent_type' not in df.columns:
        df['memory_agent_type'] = df['filename'].str.extract(r'compare_agent_([ab])_')
    
    # 添加一个新列来保存实际的记忆类型，考虑到Agent B实验中的列关系
    if 'actual_memory_type' not in df.columns:
        df['actual_memory_type'] = 'unknown'
        # 对于Agent A：记忆类型在memory_constraint_a中
        # 对于Agent B：记忆类型在memory_constraint_b中
        for idx, row in df.iterrows():
            if row['memory_agent_type'] == 'a':
                df.at[idx, 'actual_memory_type'] = row['memory_constraint_a']
            else:  # agent_type == 'b'
                df.at[idx, 'actual_memory_type'] = row['memory_constraint_b']
    
    # 设置一致的样式
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 创建token效率分析图表
    plt.figure(figsize=(16, 14))
    
    # 定义一致的颜色
    colors = {'With Memory': '#1f77b4', 'No Memory': '#ff7f0e'}
    
    # 1. Agent A: Token efficiency (tokens per game)
    plt.subplot(2, 2, 1)
    agent_a_data = df[df['memory_agent_type'] == 'a']
    if not agent_a_data.empty:
        token_data = []
        for _, row in agent_a_data.iterrows():
            token_data.append({
                'memory_type': row['actual_memory_type'],
                'board_size': row['board_size'],
                'agent_type': 'With Memory',
                'tokens_per_game': row['agent_a_tokens_per_game']
            })
            token_data.append({
                'memory_type': 'baseline',
                'board_size': row['board_size'],
                'agent_type': 'No Memory',
                'tokens_per_game': row['agent_b_tokens_per_game']
            })
        
        token_df = pd.DataFrame(token_data)
        
        # 绘制条形图，过滤掉baseline
        sns.barplot(data=token_df[token_df['memory_type'] != 'baseline'], 
                   x='memory_type', y='tokens_per_game', 
                   hue='agent_type', palette=colors, ax=plt.gca())
        
        # Add value labels
        for p in plt.gca().patches:
            height = p.get_height()
            if height > 0:
                plt.gca().text(p.get_x() + p.get_width()/2., height + 100,
                        f'{int(height)}',
                        ha="center", fontsize=10)
        
        plt.title('Agent A: Tokens per Game (Memory vs No Memory)', fontsize=14, fontweight='bold')
        plt.xlabel('Memory Type', fontsize=12)
        plt.ylabel('Tokens per Game', fontsize=12)
        
        # 简化标签设置，不过滤任何值
        plt.xticks(rotation=0)
        memory_types = sorted([t for t in token_df['memory_type'].unique() if t != 'baseline'])
        plt.gca().set_xticklabels([t.replace('_only', ' Memory').title() for t in memory_types])
    
    # 2. Agent B: Token efficiency (tokens per game)
    plt.subplot(2, 2, 2)
    agent_b_data = df[df['memory_agent_type'] == 'b']
    if not agent_b_data.empty:
        token_data = []
        for _, row in agent_b_data.iterrows():
            token_data.append({
                'memory_type': row['actual_memory_type'],
                'board_size': row['board_size'],
                'agent_type': 'With Memory',
                'tokens_per_game': row['agent_b_tokens_per_game']
            })
            token_data.append({
                'memory_type': 'baseline',
                'board_size': row['board_size'],
                'agent_type': 'No Memory',
                'tokens_per_game': row['agent_a_tokens_per_game']
            })
        
        token_df = pd.DataFrame(token_data)
        
        # 绘制条形图，过滤掉baseline
        sns.barplot(data=token_df[token_df['memory_type'] != 'baseline'], 
                   x='memory_type', y='tokens_per_game', 
                   hue='agent_type', palette=colors, ax=plt.gca())
        
        # Add value labels
        for p in plt.gca().patches:
            height = p.get_height()
            if height > 0:
                plt.gca().text(p.get_x() + p.get_width()/2., height + 100,
                        f'{int(height)}',
                        ha="center", fontsize=10)
        
        plt.title('Agent B: Tokens per Game (Memory vs No Memory)', fontsize=14, fontweight='bold')
        plt.xlabel('Memory Type', fontsize=12)
        plt.ylabel('Tokens per Game', fontsize=12)
        
        # 简化标签设置，不过滤任何值
        plt.xticks(rotation=0)
        memory_types = sorted([t for t in token_df['memory_type'].unique() if t != 'baseline'])
        plt.gca().set_xticklabels([t.replace('_only', ' Memory').title() for t in memory_types])
    
    # 3. Token efficiency by board size (Agent A)
    plt.subplot(2, 2, 3)
    if not agent_a_data.empty:
        board_token_data = []
        for _, row in agent_a_data.iterrows():
            board_token_data.append({
                'board_size': row['board_size'],
                'agent_type': 'With Memory',
                'tokens_per_game': row['agent_a_tokens_per_game']
            })
            board_token_data.append({
                'board_size': row['board_size'],
                'agent_type': 'No Memory',
                'tokens_per_game': row['agent_b_tokens_per_game']
            })
        
        board_token_df = pd.DataFrame(board_token_data)
        
        ax3 = sns.barplot(data=board_token_df, x='board_size', y='tokens_per_game', hue='agent_type', palette=colors)
        
        # Add value labels
        for p in ax3.patches:
            height = p.get_height()
            if height > 0:
                ax3.text(p.get_x() + p.get_width()/2., height + 100,
                        f'{int(height)}',
                        ha="center", fontsize=10)
        
        plt.title('Agent A: Tokens per Game by Board Size', fontsize=14, fontweight='bold')
        plt.xlabel('Board Size', fontsize=12)
        plt.ylabel('Tokens per Game', fontsize=12)
    
    # 4. Win-Token ratio (win rate per token)
    plt.subplot(2, 2, 4)
    all_data = pd.concat([agent_a_data, agent_b_data])
    if not all_data.empty:
        efficiency_data = []
        
        for _, row in all_data.iterrows():
            agent_letter = row['memory_agent_type']
            if agent_letter == 'a':
                memory_win_token_ratio = row['agent_a_win_token_ratio']
                baseline_win_token_ratio = row['agent_b_win_token_ratio']
            else:  # agent_letter == 'b'
                memory_win_token_ratio = row['agent_b_win_token_ratio']
                baseline_win_token_ratio = row['agent_a_win_token_ratio']
            
            efficiency_data.append({
                'memory_type': row['actual_memory_type'],
                'board_size': row['board_size'],
                'agent_type': 'With Memory',
                'win_token_ratio': memory_win_token_ratio
            })
            efficiency_data.append({
                'memory_type': 'baseline',
                'board_size': row['board_size'],
                'agent_type': 'No Memory',
                'win_token_ratio': baseline_win_token_ratio
            })
        
        efficiency_df = pd.DataFrame(efficiency_data)
        
        # 绘制条形图，过滤掉baseline
        sns.barplot(data=efficiency_df[efficiency_df['memory_type'] != 'baseline'], 
                   x='memory_type', y='win_token_ratio', 
                   hue='agent_type', palette=colors, ax=plt.gca())
        
        # Add value labels
        for p in plt.gca().patches:
            height = p.get_height()
            if height > 0:
                plt.gca().text(p.get_x() + p.get_width()/2., height + 0.1,
                        f'{height:.2f}',
                        ha="center", fontsize=10)
        
        plt.title('Win-Token Efficiency Ratio (All Agents)', fontsize=14, fontweight='bold')
        plt.xlabel('Memory Type', fontsize=12)
        plt.ylabel('Win Rate per 10,000 Tokens', fontsize=12)
        
        # 简化标签设置，不过滤任何值
        plt.xticks(rotation=0)
        memory_types = sorted([t for t in efficiency_df['memory_type'].unique() if t != 'baseline'])
        plt.gca().set_xticklabels([t.replace('_only', ' Memory').title() for t in memory_types])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'memory_token_efficiency.png'), dpi=300, bbox_inches='tight')
    print(f"Memory token efficiency analysis saved to {os.path.join(save_path, 'memory_token_efficiency.png')}")
    
    # 生成统计数据
    token_stats = {}
    
    # 计算Agent A的token统计数据
    if not agent_a_data.empty:
        agent_a_stats = agent_a_data.groupby(['actual_memory_type', 'board_size']).agg({
            'agent_a_tokens_per_game': ['mean', 'median', 'std'],
            'agent_b_tokens_per_game': ['mean', 'median', 'std'],
            'agent_a_win_token_ratio': ['mean', 'median', 'std'],
            'agent_b_win_token_ratio': ['mean', 'median', 'std']
        }).reset_index()
        
        token_stats['agent_a'] = agent_a_stats
    
    # 计算Agent B的token统计数据
    if not agent_b_data.empty:
        agent_b_stats = agent_b_data.groupby(['actual_memory_type', 'board_size']).agg({
            'agent_a_tokens_per_game': ['mean', 'median', 'std'],
            'agent_b_tokens_per_game': ['mean', 'median', 'std'],
            'agent_a_win_token_ratio': ['mean', 'median', 'std'],
            'agent_b_win_token_ratio': ['mean', 'median', 'std']
        }).reset_index()
        
        token_stats['agent_b'] = agent_b_stats
    
    # 保存统计数据到CSV文件
    for name, stats in token_stats.items():
        stats.to_csv(os.path.join(save_path, f'memory_token_stats_{name}.csv'))
        print(f"Token efficiency statistics for {name} saved to {os.path.join(save_path, f'memory_token_stats_{name}.csv')}")
    
    return token_stats

def analyze_memory_comparison_matrix(df=None, save_path='experiments/results/analysis/figures/memory_baseline'):
    """
    Create a matrix view comparing different agent combinations with and without memory
    
    Args:
        df: Experiment data DataFrame, loaded if None
        save_path: Path to save figures
    """
    if df is None:
        # 加载memory_comparison和baseline数据
        df = load_experiment_data(experiment_type='memory_comparison')
        
        # 如果没有加载到数据，也尝试加载constrained数据作为补充
        if df.empty:
            df = load_experiment_data(experiment_type='constrained')
    
    os.makedirs(save_path, exist_ok=True)
    
    # 检查数据是否可用
    if df.empty:
        logger.warning("No data found for memory comparison matrix analysis")
        return None
    
    # 定义agent和memory类型组合
    combinations = {
        "Agent A (Graph Memory)": {"agent": "agent_a", "memory": "graph_only"},
        "Agent A (Vector Memory)": {"agent": "agent_a", "memory": "vector_only"},
        "Agent A (No Memory)": {"agent": "agent_a", "memory": "baseline"},
        "Agent B (Graph Memory)": {"agent": "agent_b", "memory": "graph_only"},
        "Agent B (Vector Memory)": {"agent": "agent_b", "memory": "vector_only"},
        "Agent B (No Memory)": {"agent": "agent_b", "memory": "baseline"}
    }
    
    # 提取所有可用的board sizes
    board_sizes = sorted(df['board_size'].unique(), key=lambda x: int(x.split('x')[0]) if isinstance(x, str) else int(x))
    
    # 设置图表样式
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 定义颜色映射 - 从绿色(高胜率)到红色(低胜率)
    cmap = plt.cm.RdYlGn
    
    # 为每个棋盘大小创建一个矩阵视图
    for board_size in board_sizes:
        # 过滤特定棋盘大小的数据
        board_df = df[df['board_size'] == board_size]
        
        # 如果没有数据，跳过这个棋盘大小
        if board_df.empty:
            continue
        
        # 创建胜率矩阵 - 行: 主体代理, 列: 对手代理
        matrix_size = len(combinations)
        win_matrix = np.zeros((matrix_size, matrix_size))
        games_played_matrix = np.zeros((matrix_size, matrix_size))
        
        # 收集矩阵数据
        for i, (agent1_name, agent1_info) in enumerate(combinations.items()):
            for j, (agent2_name, agent2_info) in enumerate(combinations.items()):
                # 跳过自己对自己的组合
                if i == j:
                    win_matrix[i, j] = 0.5  # 自己对自己默认为50%胜率
                    games_played_matrix[i, j] = 0
                    continue
                
                # 查找匹配的实验数据
                match_rows = board_df[
                    ((board_df['memory_constraint_a'] == agent1_info['memory']) & 
                     (board_df['memory_constraint_b'] == agent2_info['memory']))
                ]
                
                if not match_rows.empty:
                    # 找到匹配的实验数据
                    row = match_rows.iloc[0]
                    
                    # 确定胜率 - 我们关注agent1(行)对agent2(列)的胜率
                    if agent1_info['agent'] == 'agent_a':
                        win_matrix[i, j] = row['agent_a_win_rate']
                    else:
                        win_matrix[i, j] = row['agent_b_win_rate']
                        
                    # 记录游戏数量
                    games_played_matrix[i, j] = row['total_games']
                
                # 如果找不到数据，尝试反向匹配(agent2 vs agent1)并转换胜率
                elif len(board_df[
                    ((board_df['memory_constraint_a'] == agent2_info['memory']) & 
                     (board_df['memory_constraint_b'] == agent1_info['memory']))
                ]) > 0:
                    # 找到反向匹配的实验数据
                    row = board_df[
                        ((board_df['memory_constraint_a'] == agent2_info['memory']) & 
                         (board_df['memory_constraint_b'] == agent1_info['memory']))
                    ].iloc[0]
                    
                    # 转换胜率 - 对手的失败率等于我方的胜率
                    if agent2_info['agent'] == 'agent_a':
                        win_matrix[i, j] = 1.0 - row['agent_a_win_rate']
                    else:
                        win_matrix[i, j] = 1.0 - row['agent_b_win_rate']
                    
                    # 记录游戏数量
                    games_played_matrix[i, j] = row['total_games']
        
        # 创建矩阵图
        plt.figure(figsize=(12, 10))
        
        # 绘制胜率热力图
        ax = plt.subplot(1, 1, 1)
        im = ax.imshow(win_matrix, cmap=cmap, vmin=0, vmax=1)
        
        # 设置坐标轴刻度和标签
        ax.set_xticks(np.arange(matrix_size))
        ax.set_yticks(np.arange(matrix_size))
        ax.set_xticklabels(list(combinations.keys()), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticklabels(list(combinations.keys()))
        
        # 添加颜色条
        cbar = plt.colorbar(im)
        cbar.set_label('Win Rate (row agent vs column agent)')
        
        # 在每个单元格添加胜率文本和游戏数量
        for i in range(matrix_size):
            for j in range(matrix_size):
                # 确定文本颜色 - 深色背景用白色文本，浅色背景用黑色文本
                text_color = "white" if 0.3 < win_matrix[i, j] < 0.7 else "black"
                
                # 添加胜率文本
                text = ax.text(j, i, f"{win_matrix[i, j]:.2f}\n({int(games_played_matrix[i, j])})",
                              ha="center", va="center", color=text_color,
                              fontweight="bold" if games_played_matrix[i, j] > 0 else "normal")
        
        # 设置图表标题和轴标签
        plt.title(f"Memory Comparison Matrix - {board_size} Board Size", fontsize=14, fontweight='bold')
        plt.xlabel("Opponent Agent", fontsize=12)
        plt.ylabel("Agent", fontsize=12)
        
        # 添加解释性文本
        plt.figtext(0.5, 0.01, 
                   "Each cell shows the win rate of the row agent against the column agent.\nNumbers in parentheses indicate games played.",
                   ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        # 紧凑布局
        plt.tight_layout()
        
        # 保存图表
        matrix_filename = f"memory_comparison_matrix_{board_size}.png"
        plt.savefig(os.path.join(save_path, matrix_filename), dpi=300, bbox_inches='tight')
        logger.info(f"Memory comparison matrix for {board_size} saved to {os.path.join(save_path, matrix_filename)}")
    
    # 为所有棋盘大小创建综合分析表
    all_comparisons = []
    
    # 遍历所有可能的代理组合
    for agent1_name, agent1_info in combinations.items():
        for agent2_name, agent2_info in combinations.items():
            # 跳过自己对自己的组合
            if agent1_name == agent2_name:
                continue
                
            # 遍历所有棋盘大小
            for board_size in board_sizes:
                # 查找匹配的实验数据
                board_df = df[df['board_size'] == board_size]
                
                # 直接匹配
                match_rows = board_df[
                    ((board_df['memory_constraint_a'] == agent1_info['memory']) & 
                     (board_df['memory_constraint_b'] == agent2_info['memory']))
                ]
                
                if not match_rows.empty:
                    # 找到匹配的实验数据
                    row = match_rows.iloc[0]
                    
                    # 创建比较条目
                    comparison = {
                        'board_size': board_size,
                        'agent1': agent1_name,
                        'agent2': agent2_name,
                        'total_games': row['total_games']
                    }
                    
                    # 设置胜率
                    if agent1_info['agent'] == 'agent_a':
                        comparison['win_rate'] = row['agent_a_win_rate']
                        comparison['agent1_tokens'] = row['agent_a_total_tokens'] / row['total_games'] if row['total_games'] > 0 else 0
                        comparison['agent2_tokens'] = row['agent_b_total_tokens'] / row['total_games'] if row['total_games'] > 0 else 0
                    else:
                        comparison['win_rate'] = row['agent_b_win_rate']
                        comparison['agent1_tokens'] = row['agent_b_total_tokens'] / row['total_games'] if row['total_games'] > 0 else 0
                        comparison['agent2_tokens'] = row['agent_a_total_tokens'] / row['total_games'] if row['total_games'] > 0 else 0
                    
                    all_comparisons.append(comparison)
                
                # 反向匹配
                reverse_match_rows = board_df[
                    ((board_df['memory_constraint_a'] == agent2_info['memory']) & 
                     (board_df['memory_constraint_b'] == agent1_info['memory']))
                ]
                
                if not match_rows.empty and reverse_match_rows.empty and not match_rows.empty:
                    # 找到反向匹配的实验数据
                    row = reverse_match_rows.iloc[0]
                    
                    # 创建比较条目
                    comparison = {
                        'board_size': board_size,
                        'agent1': agent1_name,
                        'agent2': agent2_name,
                        'total_games': row['total_games']
                    }
                    
                    # 设置胜率 (反向转换)
                    if agent2_info['agent'] == 'agent_a':
                        comparison['win_rate'] = 1.0 - row['agent_a_win_rate']
                        comparison['agent1_tokens'] = row['agent_b_total_tokens'] / row['total_games'] if row['total_games'] > 0 else 0
                        comparison['agent2_tokens'] = row['agent_a_total_tokens'] / row['total_games'] if row['total_games'] > 0 else 0
                    else:
                        comparison['win_rate'] = 1.0 - row['agent_b_win_rate']
                        comparison['agent1_tokens'] = row['agent_a_total_tokens'] / row['total_games'] if row['total_games'] > 0 else 0
                        comparison['agent2_tokens'] = row['agent_b_total_tokens'] / row['total_games'] if row['total_games'] > 0 else 0
                    
                    all_comparisons.append(comparison)
    
    # 将所有比较结果转换为DataFrame并保存
    if all_comparisons:
        all_comparisons_df = pd.DataFrame(all_comparisons)
        all_comparisons_df.to_csv(os.path.join(save_path, 'memory_comparison_matrix_data.csv'))
        logger.info(f"Memory comparison matrix data saved to {os.path.join(save_path, 'memory_comparison_matrix_data.csv')}")
        return all_comparisons_df
    
    return None

def analyze_decision_time(df=None, game_logs_df=None, save_path='experiments/results/analysis/figures/memory_baseline'):
    """
    Analyze decision time differences between agents with and without memory in baseline experiments
    
    Args:
        df: DataFrame containing experiment data
        game_logs_df: DataFrame containing game log data with response times
        save_path: Path to save figures
    """
    # 加载必要的数据
    if df is None:
        df = load_experiment_data(experiment_type='memory_comparison')
    
    if game_logs_df is None:
        # 尝试加载包含决策时间的游戏日志数据
        game_logs_df = process_game_logs(experiment_type='memory_comparison')
    
    os.makedirs(save_path, exist_ok=True)
    
    # 检查数据是否可用
    if df.empty:
        print("No memory comparison data found")
        return None
    
    # 检查是否有决策时间数据
    if game_logs_df.empty or 'response_time' not in game_logs_df.columns:
        print("No decision time data found in game logs")
        return None
    
    print("Analyzing decision time differences between memory and non-memory agents...")
    
    # 设置图表样式
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 创建多图布局
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. 平均决策时间 - 记忆代理 vs 非记忆代理
    ax1 = axes[0, 0]
    
    # 按代理ID和记忆约束分组计算平均决策时间
    avg_decision_time = game_logs_df.groupby(['agent_id', 'memory_constraint_a'])['response_time'].mean().reset_index()
    
    if not avg_decision_time.empty:
        # 确保记忆约束标签格式一致
        avg_decision_time['memory_type'] = avg_decision_time['memory_constraint_a'].apply(
            lambda x: 'No Memory' if x == 'baseline' else 'With Memory'
        )
        
        # 绘制平均决策时间条形图
        sns.barplot(data=avg_decision_time, x='agent_id', y='response_time', hue='memory_type', ax=ax1)
        
        ax1.set_title('Average Decision Time: Memory vs Non-Memory Agents', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Agent', fontsize=12)
        ax1.set_ylabel('Average Decision Time (seconds)', fontsize=12)
        ax1.legend(title='Memory Type')
        
        # 添加数值标签
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%.2f s')
    else:
        ax1.text(0.5, 0.5, 'No average decision time data available', ha='center', va='center', transform=ax1.transAxes)
    
    # 2. 决策时间随棋盘大小的变化
    ax2 = axes[0, 1]
    
    # 按棋盘大小和记忆约束分组计算平均决策时间
    board_decision_time = game_logs_df.groupby(['board_size', 'memory_constraint_a'])['response_time'].mean().reset_index()
    
    if not board_decision_time.empty:
        # 确保记忆约束标签格式一致
        board_decision_time['memory_type'] = board_decision_time['memory_constraint_a'].apply(
            lambda x: 'No Memory' if x == 'baseline' else 'With Memory'
        )
        
        # 确保棋盘大小排序正确
        board_decision_time['board_size_num'] = board_decision_time['board_size'].astype(str).str.extract('(\d+)').astype(int)
        board_decision_time = board_decision_time.sort_values('board_size_num')
        
        # 绘制决策时间随棋盘大小的变化
        sns.lineplot(data=board_decision_time, x='board_size', y='response_time', hue='memory_type', 
                    marker='o', markersize=10, linewidth=2, ax=ax2)
        
        ax2.set_title('Decision Time by Board Size: Memory vs Non-Memory', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Board Size', fontsize=12)
        ax2.set_ylabel('Average Decision Time (seconds)', fontsize=12)
        ax2.legend(title='Memory Type')
        
        # 添加数值标签
        for i, row in board_decision_time.iterrows():
            ax2.annotate(f"{row['response_time']:.2f}s", 
                        (row['board_size'], row['response_time']),
                        textcoords="offset points", xytext=(0,10), ha='center')
    else:
        ax2.text(0.5, 0.5, 'No board size decision time data available', ha='center', va='center', transform=ax2.transAxes)
    
    # 3. 决策时间随游戏进程的变化
    ax3 = axes[1, 0]
    
    if 'turn' in game_logs_df.columns:
        # 计算每个游戏的总回合数
        game_turns = game_logs_df.groupby(['run_id', 'game_id'])['turn'].max() + 1
        
        # 创建一个新的DataFrame，包含游戏进程百分比和决策时间
        game_progress = []
        
        for _, row in game_logs_df.iterrows():
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
            
            game_progress.append({
                'agent_id': row['agent_id'],
                'memory_constraint': row['memory_constraint_a'],
                'response_time': row['response_time'],
                'turn_percentage': turn_percentage,
                'board_size': row['board_size']
            })
        
        if game_progress:
            # 转换为DataFrame
            progress_df = pd.DataFrame(game_progress)
            
            # 创建游戏进度区间
            progress_df['progress_bin'] = pd.cut(
                progress_df['turn_percentage'],
                bins=[0, 25, 50, 75, 100],
                labels=['0-25%', '25-50%', '50-75%', '75-100%']
            )
            
            # 确保记忆约束标签格式一致
            progress_df['memory_type'] = progress_df['memory_constraint'].apply(
                lambda x: 'No Memory' if x == 'baseline' else 'With Memory'
            )
            
            # 按游戏进度和记忆类型分组计算平均决策时间
            progress_time = progress_df.groupby(['progress_bin', 'memory_type'])['response_time'].mean().reset_index()
            
            # 绘制决策时间随游戏进程的变化
            sns.lineplot(data=progress_time, x='progress_bin', y='response_time', hue='memory_type',
                        marker='o', markersize=10, linewidth=2, ax=ax3)
            
            ax3.set_title('Decision Time Throughout Game Progress', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Game Progress', fontsize=12)
            ax3.set_ylabel('Average Decision Time (seconds)', fontsize=12)
            ax3.legend(title='Memory Type')
            
            # 添加数值标签
            for i, row in progress_time.iterrows():
                ax3.annotate(f"{row['response_time']:.2f}s", 
                            (row['progress_bin'], row['response_time']),
                            textcoords="offset points", xytext=(0,10), ha='center')
        else:
            ax3.text(0.5, 0.5, 'No game progress data available', ha='center', va='center', transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, 'No turn data available', ha='center', va='center', transform=ax3.transAxes)
    
    # 4. 决策时间与操作复杂性的关系
    ax4 = axes[1, 1]
    
    if 'operation' in game_logs_df.columns:
        # 按记忆操作类型和记忆约束分组计算平均决策时间
        operation_time = game_logs_df.groupby(['operation', 'memory_constraint_a'])['response_time'].mean().reset_index()
        
        if not operation_time.empty:
            # 确保记忆约束标签格式一致
            operation_time['memory_type'] = operation_time['memory_constraint_a'].apply(
                lambda x: 'No Memory' if x == 'baseline' else 'With Memory'
            )
            
            # 绘制不同操作类型的决策时间条形图
            sns.barplot(data=operation_time, x='operation', y='response_time', hue='memory_type', ax=ax4)
            
            ax4.set_title('Decision Time by Memory Operation Type', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Operation Type', fontsize=12)
            ax4.set_ylabel('Average Decision Time (seconds)', fontsize=12)
            ax4.legend(title='Memory Type')
            
            # 添加数值标签
            for container in ax4.containers:
                ax4.bar_label(container, fmt='%.2f s')
        else:
            ax4.text(0.5, 0.5, 'No operation decision time data available', ha='center', va='center', transform=ax4.transAxes)
    else:
        ax4.text(0.5, 0.5, 'No operation data available', ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'decision_time_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Decision time analysis saved to {os.path.join(save_path, 'decision_time_analysis.png')}")
    
    # 计算决策时间相关统计数据
    decision_stats = {}
    
    # 整体统计
    if 'response_time' in game_logs_df.columns:
        overall_stats = game_logs_df.groupby('memory_constraint_a')['response_time'].agg(['mean', 'median', 'std', 'min', 'max']).reset_index()
        overall_stats['memory_type'] = overall_stats['memory_constraint_a'].apply(
            lambda x: 'No Memory' if x == 'baseline' else 'With Memory'
        )
        decision_stats['overall'] = overall_stats
    
    # 按棋盘大小统计
    if 'board_size' in game_logs_df.columns:
        board_stats = game_logs_df.groupby(['board_size', 'memory_constraint_a'])['response_time'].agg(['mean', 'median', 'std']).reset_index()
        board_stats['memory_type'] = board_stats['memory_constraint_a'].apply(
            lambda x: 'No Memory' if x == 'baseline' else 'With Memory'
        )
        decision_stats['board_size'] = board_stats
    
    # 按代理统计
    if 'agent_id' in game_logs_df.columns:
        agent_stats = game_logs_df.groupby(['agent_id', 'memory_constraint_a'])['response_time'].agg(['mean', 'median', 'std']).reset_index()
        agent_stats['memory_type'] = agent_stats['memory_constraint_a'].apply(
            lambda x: 'No Memory' if x == 'baseline' else 'With Memory'
        )
        decision_stats['agent'] = agent_stats
    
    # 按操作类型统计
    if 'operation' in game_logs_df.columns:
        operation_stats = game_logs_df.groupby(['operation', 'memory_constraint_a'])['response_time'].agg(['mean', 'median', 'std']).reset_index()
        operation_stats['memory_type'] = operation_stats['memory_constraint_a'].apply(
            lambda x: 'No Memory' if x == 'baseline' else 'With Memory'
        )
        decision_stats['operation'] = operation_stats
    
    # 保存统计数据到CSV文件
    for name, stats in decision_stats.items():
        stats.to_csv(os.path.join(save_path, f'decision_time_{name}_stats.csv'))
        print(f"Decision time statistics saved to {os.path.join(save_path, f'decision_time_{name}_stats.csv')}")
    
    return decision_stats

def analyze_baseline_win_rates(df=None, save_path='experiments/results/analysis/figures/memory_baseline'):
    """
    Analyze win rates for baseline experiments (no memory agents only)
    
    Args:
        df: Memory baseline DataFrame, loaded if None (not used if read_from_dir=True)
        save_path: Path to save figures
    
    Returns:
        DataFrame with baseline win rates
    """
    print("Analyzing baseline (no memory) win rates...")
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 直接从baseline目录读取数据
    baseline_data = []
    baseline_dirs = [
        "results/memory_baseline/baseline_nomem_agent_avsagent_b_b3x3_20250426_154821/baseline_b3x3_20250426_154821",
        "results/memory_baseline/baseline_nomem_agent_avsagent_b_b9x9_20250426_172255/baseline_b9x9_20250426_172255"
    ]
    
    for dir_path in baseline_dirs:
        # 检查目录是否存在
        if not os.path.exists(dir_path):
            print(f"Warning: Directory {dir_path} does not exist")
            continue
            
        # 读取配置文件
        config_path = os.path.join(dir_path, "config.json")
        if not os.path.exists(config_path):
            print(f"Warning: Config file not found in {dir_path}")
            continue
            
        with open(config_path, "r") as f:
            config = json.load(f)
            
        # 读取统计文件
        stats_path = os.path.join(dir_path, "stats.json")
        if not os.path.exists(stats_path):
            print(f"Warning: Stats file not found in {dir_path}")
            continue
            
        with open(stats_path, "r") as f:
            stats = json.load(f)
            
        # 计算胜率
        total_completed_games = stats["wins_a"] + stats["wins_b"] + stats["draws"]
        if total_completed_games == 0:
            print(f"Warning: No completed games found in {dir_path}")
            continue
            
        # 添加到数据集
        baseline_data.append({
            'board_size': config["board_size"],
            'agent_a_wins': stats["wins_a"],
            'agent_b_wins': stats["wins_b"],
            'draws': stats["draws"],
            'total_games': total_completed_games,
            'agent_a_win_rate': stats["wins_a"] / total_completed_games,
            'agent_b_win_rate': stats["wins_b"] / total_completed_games,
            'draw_rate': stats["draws"] / total_completed_games,
            'timeouts': stats.get("timeouts", 0),
            'errors': stats.get("errors", 0),
            'model': config.get("model", "unknown")
        })
    
    # 如果没有找到数据，尝试使用传入的df
    if not baseline_data and df is not None:
        print("No baseline data found in directory, using provided DataFrame")
        baseline_df = df[
            (df['memory_constraint_a'] == 'baseline') & 
            (df['memory_constraint_b'] == 'baseline')
        ]
        if not baseline_df.empty:
            for _, row in baseline_df.iterrows():
                baseline_data.append({
                    'board_size': row['board_size'],
                    'agent_a_wins': row.get('agent_a_total_wins', 0),
                    'agent_b_wins': row.get('agent_b_total_wins', 0),
                    'draws': row.get('draws', 0),
                    'total_games': row.get('total_games', 0),
                    'agent_a_win_rate': row.get('agent_a_win_rate', 0),
                    'agent_b_win_rate': row.get('agent_b_win_rate', 0),
                    'draw_rate': row.get('draw_rate', 0),
                    'model': row.get('model', 'unknown')
                })
    
    if not baseline_data:
        print("No baseline data found.")
        return None
        
    # 转换为DataFrame
    baseline_df = pd.DataFrame(baseline_data)
    
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 绘制胜率条形图
    plt.figure(figsize=(10, 6))
    
    # 准备绘图数据
    win_data = []
    for _, row in baseline_df.iterrows():
        board_size = str(row['board_size'])
        win_data.append({
            'Agent': 'Agent A',
            'Board Size': f"{board_size}x{board_size}",
            'Win Rate (%)': row['agent_a_win_rate'] * 100
        })
        win_data.append({
            'Agent': 'Agent B',
            'Board Size': f"{board_size}x{board_size}",
            'Win Rate (%)': row['agent_b_win_rate'] * 100
        })
    
    win_df = pd.DataFrame(win_data)
    
    # 按棋盘大小排序
    board_sizes = sorted(win_df['Board Size'].unique(), 
                        key=lambda x: int(x.split('x')[0]))
    
    # 创建条形图
    ax = sns.barplot(
        data=win_df, 
        x='Board Size', 
        y='Win Rate (%)', 
        hue='Agent',
        palette={'Agent A': '#1f77b4', 'Agent B': '#ff7f0e'},
        order=board_sizes
    )
    
    # 在每个条形上添加数值标签
    for p in ax.patches:
        height = p.get_height()
        if height > 0.5:  # 只为有意义的高度添加标签
            ax.text(
                p.get_x() + p.get_width()/2.,
                height + 1,
                f'{height:.1f}%',
                ha="center",
                fontsize=10
            )
    
    # 添加50%基准线
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
    
    # 增强绘图标题和标签
    plt.title('Baseline Win Rates: Agent A vs Agent B (No Memory)', 
            fontsize=16, fontweight='bold')
    plt.xlabel('Board Size', fontsize=14)
    plt.ylabel('Win Rate (%)', fontsize=14)
    plt.legend(title='Agent', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # 添加说明文字
    plt.figtext(
        0.5, 0.01, 
        "Comparison of win rates when neither agent has memory capabilities",
        ha='center', fontsize=11, style='italic'
    )
    
    # 保存图表
    output_path = os.path.join(save_path, 'baseline_win_rates.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Baseline win rates analysis saved to {output_path}")
    
    # 添加超时和错误信息的条形图
    plt.figure(figsize=(10, 6))
    
    # 准备数据
    completion_data = []
    for _, row in baseline_df.iterrows():
        board_size = str(row['board_size'])
        total_attempts = row['total_games'] + row.get('timeouts', 0) + row.get('errors', 0)
        
        # 计算完成率和超时率
        completion_rate = row['total_games'] / total_attempts * 100 if total_attempts > 0 else 0
        timeout_rate = row.get('timeouts', 0) / total_attempts * 100 if total_attempts > 0 else 0
        error_rate = row.get('errors', 0) / total_attempts * 100 if total_attempts > 0 else 0
        
        completion_data.append({
            'Board Size': f"{board_size}x{board_size}",
            'Rate Type': 'Completion Rate',
            'Percentage': completion_rate
        })
        completion_data.append({
            'Board Size': f"{board_size}x{board_size}",
            'Rate Type': 'Timeout Rate',
            'Percentage': timeout_rate
        })
        completion_data.append({
            'Board Size': f"{board_size}x{board_size}",
            'Rate Type': 'Error Rate',
            'Percentage': error_rate
        })
    
    completion_df = pd.DataFrame(completion_data)
    
    # 创建条形图
    ax2 = sns.barplot(
        data=completion_df,
        x='Board Size',
        y='Percentage',
        hue='Rate Type',
        palette={'Completion Rate': '#2ca02c', 'Timeout Rate': '#d62728', 'Error Rate': '#9467bd'},
        order=board_sizes
    )
    
    # 添加数值标签
    for p in ax2.patches:
        height = p.get_height()
        if height > 0.5:
            ax2.text(
                p.get_x() + p.get_width()/2.,
                height + 1,
                f'{height:.1f}%',
                ha="center",
                fontsize=9
            )
    
    # 设置标题和标签
    plt.title('Game Completion, Timeout, and Error Rates (No Memory)', fontsize=16, fontweight='bold')
    plt.xlabel('Board Size', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.legend(title='Rate Type', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # 保存图表
    completion_path = os.path.join(save_path, 'baseline_completion_rates.png')
    plt.tight_layout()
    plt.savefig(completion_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Baseline completion rates analysis saved to {completion_path}")
    
    # 保存统计数据到CSV文件
    stats_path = os.path.join(save_path, 'baseline_win_rates_stats.csv')
    baseline_df.to_csv(stats_path, index=False)
    print(f"Baseline statistics saved to {stats_path}")
    
    return baseline_df

if __name__ == "__main__":
    # Create directory for saving figures
    os.makedirs('experiments/results/analysis/figures', exist_ok=True)
    os.makedirs('experiments/results/analysis/figures/memory_baseline', exist_ok=True)
    
    # Load constrained data
    print("Loading constrained memory experiment data...")
    df_constrained = load_experiment_data(experiment_type='constrained')
    
    # Run analyses for constrained data
    analyze_win_rates(df_constrained)
    analyze_token_efficiency(df_constrained)
    analyze_performance_retention(df_constrained)
    analyze_memory_calls(df_constrained)
    
    # Try to load memory comparison data
    print("Loading memory comparison experiment data...")
    df_memory_comparison = load_experiment_data(experiment_type='memory_comparison')
    
    # Check if memory comparison data exists
    if not df_memory_comparison.empty:
        try:
            print("Running memory baseline analyses...")
            # 直接从文件名中提取agent类型信息，不再检查'memory_agent_type'列
            analyze_memory_baseline_comparison(df_memory_comparison)
            analyze_baseline_win_rates(df_memory_comparison)  # 添加新函数的调用
            analyze_memory_token_efficiency(df_memory_comparison)
            
            # 注意：还需要更新其他两个函数
            # analyze_memory_comparison_matrix(df_memory_comparison)
            # analyze_decision_time(df_memory_comparison)
            print("Memory baseline analysis completed.")
        except Exception as e:
            print(f"Error in memory comparison analyses: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No memory comparison data found. Skipping memory baseline analyses.")
    
    print("Performance metrics analysis completed.") 