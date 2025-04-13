import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_experiment_data

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
    
    # Create win rate analysis charts
    plt.figure(figsize=(15, 12))
    
    # 1. Win rates by memory architecture
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
    sns.barplot(data=win_df, x='memory_constraint', y='win_rate', hue='agent')
    plt.title('Win Rates by Memory Architecture and Agent')
    plt.xlabel('Memory Architecture')
    plt.ylabel('Win Rate (%)')
    
    # 2. Win rates by board size
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
    sns.barplot(data=size_df, x='board_size', y='win_rate', hue='agent')
    plt.title('Win Rates by Board Size')
    plt.xlabel('Board Size')
    plt.ylabel('Win Rate (%)')
    
    # 3. Draw rates by memory architecture and board size
    plt.subplot(2, 2, 3)
    draw_data = []
    
    for _, row in df.iterrows():
        draw_data.append({
            'memory_constraint': row['memory_constraint_a'],
            'board_size': row['board_size'],
            'draw_rate': row['draw_rate'] * 100
        })
    
    draw_df = pd.DataFrame(draw_data)
    sns.barplot(data=draw_df, x='memory_constraint', y='draw_rate', hue='board_size')
    plt.title('Draw Rates by Memory Architecture and Board Size')
    plt.xlabel('Memory Architecture')
    plt.ylabel('Draw Rate (%)')
    
    # 4. First-player advantage by memory architecture and board size
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
        sns.barplot(data=advantage_df, x='memory_constraint', y='first_player_advantage', hue='board_size')
        plt.title('First-Player Advantage by Memory Architecture and Board Size')
        plt.xlabel('Memory Architecture')
        plt.ylabel('First-Player Advantage (deviation from 50%)')
        plt.axhline(y=0, color='red', linestyle='--')
    else:
        plt.text(0.5, 0.5, 'No first-player advantage data available', 
                 horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'win_rate_analysis.png'))
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
        sns.barplot(data=token_df, x='memory_constraint', y='tokens_per_game', hue='agent')
        plt.title('Average Token Usage per Game by Memory Architecture')
        plt.xlabel('Memory Architecture')
        plt.ylabel('Tokens per Game')
    else:
        plt.text(0.5, 0.5, 'No token usage data available', 
                 horizontalalignment='center', verticalalignment='center')
    
    # 2. Average token usage per game by board size
    plt.subplot(2, 2, 2)
    
    if token_data:
        sns.barplot(data=token_df, x='board_size', y='tokens_per_game', hue='agent')
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
        sns.barplot(data=ratio_df, x='memory_constraint', y='win_token_ratio', hue='agent')
        plt.title('Win-Rate to Token Ratio by Memory Architecture (higher is better)')
        plt.xlabel('Memory Architecture')
        plt.ylabel('Win Rate * 10000 / Tokens per Game')
    else:
        plt.text(0.5, 0.5, 'No token ratio data available', 
                 horizontalalignment='center', verticalalignment='center')
    
    # 4. Win-rate to token ratio by board size
    plt.subplot(2, 2, 4)
    
    if ratio_data:
        sns.barplot(data=ratio_df, x='board_size', y='win_token_ratio', hue='agent')
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
    
    for constraint in sorted(retention_df_a['memory_constraint'].unique()):
        data = retention_df_a[retention_df_a['memory_constraint'] == constraint]
        plt.plot(data['board_size'], data['retention_rate'], marker='o', label=f"{constraint}")
    
    plt.axhline(y=100, color='gray', linestyle='--')
    plt.title('Agent A: Performance Retention Rate by Board Size')
    plt.xlabel('Board Size')
    plt.ylabel('Performance Retention (% of 3x3 baseline)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
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
    
    for constraint in sorted(retention_df_b['memory_constraint'].unique()):
        data = retention_df_b[retention_df_b['memory_constraint'] == constraint]
        plt.plot(data['board_size'], data['retention_rate'], marker='o', label=f"{constraint}")
    
    plt.axhline(y=100, color='gray', linestyle='--')
    plt.title('Agent B: Performance Retention Rate by Board Size')
    plt.xlabel('Board Size')
    plt.ylabel('Performance Retention (% of 3x3 baseline)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
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
        sns.barplot(data=combined_df, x='board_size', y='retention_rate', hue='agent')
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
        sns.barplot(data=arch_df, x='memory_constraint', y='avg_retention', hue='board_size')
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
    retention_stats = pd.DataFrame({
        'agent_a_retention': retention_df_a.pivot(index='memory_constraint', columns='board_size', values='retention_rate'),
        'agent_b_retention': retention_df_b.pivot(index='memory_constraint', columns='board_size', values='retention_rate')
    })
    
    # Save statistics to CSV
    retention_stats.to_csv(os.path.join(save_path, 'performance_retention_statistics.csv'))
    print(f"Performance retention statistics saved to {os.path.join(save_path, 'performance_retention_statistics.csv')}")
    
    return retention_stats

if __name__ == "__main__":
    # Create directory for saving figures
    os.makedirs('experiments/results/analysis/figures', exist_ok=True)
    
    # Load data
    df = load_experiment_data(experiment_type='constrained')
    
    # Run analyses
    analyze_win_rates(df)
    analyze_token_efficiency(df)
    analyze_performance_retention(df)
    
    print("Performance metrics analysis completed.") 