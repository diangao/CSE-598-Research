import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from pathlib import Path
import re

# Get absolute path to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

def load_baseline_data():
    """
    Load baseline experiment data
    Returns a DataFrame containing all baseline experiment results
    """
    baseline_dir = os.path.join(PROJECT_ROOT, "experiments/results/baseline")
    
    # Find all experiment directories (excluding summary files)
    experiment_dirs = []
    for item in os.listdir(baseline_dir):
        if os.path.isdir(os.path.join(baseline_dir, item)) and item.startswith("baseline_b"):
            experiment_dirs.append(os.path.join(baseline_dir, item))
    
    data = []
    for exp_dir in experiment_dirs:
        # Read configuration file
        config_path = os.path.join(exp_dir, "config.json")
        if not os.path.exists(config_path):
            continue
            
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Read statistics file
        stats_path = os.path.join(exp_dir, "stats.json")
        if not os.path.exists(stats_path):
            continue
            
        with open(stats_path, "r") as f:
            stats = json.load(f)
        
        # Extract key information
        exp_data = {
            "experiment_type": "baseline",
            "board_size": config["board_size"],
            "model": config["model"],
            "games_count": config["games"],
            "agent_a_wins": stats["wins_a"],
            "agent_b_wins": stats["wins_b"],
            "draws": stats["draws"],
            "agent_a_win_rate": stats["wins_a"] / config["games"] if config["games"] > 0 else 0,
            "agent_b_win_rate": stats["wins_b"] / config["games"] if config["games"] > 0 else 0,
            "draw_rate": stats["draws"] / config["games"] if config["games"] > 0 else 0,
            "avg_moves_per_game": stats["total_moves"] / config["games"] if config["games"] > 0 else 0,
            "avg_time_per_game": stats["total_time"] / config["games"] if config["games"] > 0 else 0,
            "memory_constraint_a": "none",  # baseline experiments have no memory
            "memory_constraint_b": "none",  # baseline experiments have no memory
            "exp_id": config["exp_id"]
        }
        
        data.append(exp_data)
    
    return pd.DataFrame(data)

def load_other_experiments_data():
    """
    Load constrained and adaptive experiment data
    Returns a DataFrame containing these experiment results
    """
    from experiments.results.analysis.data_loader import load_experiment_data
    
    # Load constrained and adaptive experiment data
    df = load_experiment_data()
    
    return df

def compare_win_rates(baseline_df, other_df, save_path='experiments/results/analysis/figures/baseline_comparison'):
    """
    Compare win rates between baseline and other experiments
    
    Args:
        baseline_df: Baseline experiment data
        other_df: Other experiment (constrained and adaptive) data
        save_path: Path to save figures
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Set consistent style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Prepare data
    baseline_data = []
    for _, row in baseline_df.iterrows():
        baseline_data.append({
            'experiment_type': 'Baseline (No Memory)',
            'board_size': str(row['board_size']),
            'agent': 'Agent A',
            'win_rate': row['agent_a_win_rate'] * 100
        })
        baseline_data.append({
            'experiment_type': 'Baseline (No Memory)',
            'board_size': str(row['board_size']),
            'agent': 'Agent B',
            'win_rate': row['agent_b_win_rate'] * 100
        })
    
    # Process constrained experiment data (using only graph_only and vector_only)
    constrained_data = []
    for _, row in other_df.iterrows():
        if row['memory_constraint_a'] in ['graph_only', 'vector_only']:
            experiment_type = f"{row['memory_constraint_a'].replace('_only', ' Memory')}"
            constrained_data.append({
                'experiment_type': experiment_type,
                'board_size': str(row['board_size']),
                'agent': 'Agent A',
                'win_rate': row['agent_a_win_rate'] * 100
            })
            constrained_data.append({
                'experiment_type': experiment_type,
                'board_size': str(row['board_size']),
                'agent': 'Agent B',
                'win_rate': row['agent_b_win_rate'] * 100
            })
    
    # Merge data
    all_data = pd.DataFrame(baseline_data + constrained_data)
    
    # Create win rate comparison chart
    plt.figure(figsize=(15, 10))
    
    # Compare by board size
    board_sizes = sorted(all_data['board_size'].unique())
    for i, size in enumerate(board_sizes):
        plt.subplot(1, len(board_sizes), i+1)
        
        # Filter data for this board size
        size_data = all_data[all_data['board_size'] == size]
        
        # Create bar chart
        ax = sns.barplot(data=size_data, x='experiment_type', y='win_rate', hue='agent', palette=['#1f77b4', '#ff7f0e'])
        
        # Add value labels
        for p in ax.patches:
            height = p.get_height()
            if height > 0.5:
                ax.text(p.get_x() + p.get_width()/2., height + 1,
                      f'{height:.1f}%',
                      ha="center", fontsize=9)
        
        # Set title and labels
        plt.title(f'{size}x{size} Board Win Rates', fontsize=14, fontweight='bold')
        plt.xlabel('Memory Type', fontsize=12)
        plt.ylabel('Win Rate (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add 50% baseline
        plt.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
        
        # Adjust legend
        plt.legend(title='Agent')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'win_rates_by_board_size.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create overall win rate comparison chart
    plt.figure(figsize=(12, 8))
    
    # Calculate average win rates by experiment type
    avg_win_rates = all_data.groupby(['experiment_type', 'agent'])['win_rate'].mean().reset_index()
    
    # Create bar chart
    ax = sns.barplot(data=avg_win_rates, x='experiment_type', y='win_rate', hue='agent', palette=['#1f77b4', '#ff7f0e'])
    
    # Add value labels
    for p in ax.patches:
        height = p.get_height()
        if height > 0.5:
            ax.text(p.get_x() + p.get_width()/2., height + 1,
                  f'{height:.1f}%',
                  ha="center", fontsize=10)
    
    # Set title and labels
    plt.title('Average Win Rates by Memory Type', fontsize=16, fontweight='bold')
    plt.xlabel('Memory Type', fontsize=14)
    plt.ylabel('Win Rate (%)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add 50% baseline
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'avg_win_rates.png'), dpi=300, bbox_inches='tight')
    plt.close()

def compare_game_efficiency(baseline_df, other_df, save_path='experiments/results/analysis/figures/baseline_comparison'):
    """
    Compare game efficiency (average moves and time per game) between baseline and other experiments
    
    Args:
        baseline_df: Baseline experiment data
        other_df: Other experiment (constrained and adaptive) data
        save_path: Path to save figures
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Set consistent style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Prepare data
    baseline_data = []
    for _, row in baseline_df.iterrows():
        baseline_data.append({
            'experiment_type': 'Baseline (No Memory)',
            'board_size': str(row['board_size']),
            'avg_moves': row['avg_moves_per_game'],
            'avg_time': row['avg_time_per_game']
        })
    
    # Process constrained experiment data
    constrained_data = []
    for _, row in other_df.iterrows():
        if row['memory_constraint_a'] in ['graph_only', 'vector_only']:
            experiment_type = f"{row['memory_constraint_a'].replace('_only', ' Memory')}"
            if 'avg_moves_per_game' in row and 'avg_time_per_game' in row:
                constrained_data.append({
                    'experiment_type': experiment_type,
                    'board_size': str(row['board_size']),
                    'avg_moves': row['avg_moves_per_game'],
                    'avg_time': row['avg_time_per_game']
                })
    
    # Merge data
    all_data = pd.DataFrame(baseline_data + constrained_data)
    
    # Create average moves comparison chart
    plt.figure(figsize=(15, 10))
    
    # Compare by board size
    board_sizes = sorted(all_data['board_size'].unique())
    for i, size in enumerate(board_sizes):
        plt.subplot(1, len(board_sizes), i+1)
        
        # Filter data for this board size
        size_data = all_data[all_data['board_size'] == size]
        
        # Create bar chart
        ax = sns.barplot(data=size_data, x='experiment_type', y='avg_moves', palette='Blues_d')
        
        # Add value labels
        for p in ax.patches:
            height = p.get_height()
            if height > 0.5:
                ax.text(p.get_x() + p.get_width()/2., height + 0.5,
                      f'{height:.1f}',
                      ha="center", fontsize=9)
        
        # Set title and labels
        plt.title(f'{size}x{size} Board Average Moves Per Game', fontsize=14, fontweight='bold')
        plt.xlabel('Memory Type', fontsize=12)
        plt.ylabel('Average Moves', fontsize=12)
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'avg_moves_by_board_size.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create average game time comparison chart
    plt.figure(figsize=(15, 10))
    
    # Compare by board size
    for i, size in enumerate(board_sizes):
        plt.subplot(1, len(board_sizes), i+1)
        
        # Filter data for this board size
        size_data = all_data[all_data['board_size'] == size]
        
        # Create bar chart
        ax = sns.barplot(data=size_data, x='experiment_type', y='avg_time', palette='Greens_d')
        
        # Add value labels
        for p in ax.patches:
            height = p.get_height()
            if height > 0.5:
                ax.text(p.get_x() + p.get_width()/2., height + 1,
                      f'{height:.1f}s',
                      ha="center", fontsize=9)
        
        # Set title and labels
        plt.title(f'{size}x{size} Board Average Time Per Game', fontsize=14, fontweight='bold')
        plt.xlabel('Memory Type', fontsize=12)
        plt.ylabel('Average Time (seconds)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'avg_time_by_board_size.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create overall efficiency comparison chart
    plt.figure(figsize=(12, 10))
    
    # Average moves
    plt.subplot(2, 1, 1)
    avg_data = all_data.groupby(['experiment_type', 'board_size'])['avg_moves'].mean().reset_index()
    
    # Create line chart
    sns.lineplot(data=avg_data, x='board_size', y='avg_moves', hue='experiment_type', marker='o', linewidth=2.5)
    
    # Set title and labels
    plt.title('Average Moves Per Game by Board Size', fontsize=16, fontweight='bold')
    plt.xlabel('Board Size', fontsize=14)
    plt.ylabel('Average Moves', fontsize=14)
    plt.legend(title='Memory Type', fontsize=12)
    
    # Average game time
    plt.subplot(2, 1, 2)
    avg_time_data = all_data.groupby(['experiment_type', 'board_size'])['avg_time'].mean().reset_index()
    
    # Create line chart
    sns.lineplot(data=avg_time_data, x='board_size', y='avg_time', hue='experiment_type', marker='o', linewidth=2.5)
    
    # Set title and labels
    plt.title('Average Time Per Game by Board Size', fontsize=16, fontweight='bold')
    plt.xlabel('Board Size', fontsize=14)
    plt.ylabel('Average Time (seconds)', fontsize=14)
    plt.legend(title='Memory Type', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'efficiency_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_table(baseline_df, other_df, save_path='experiments/results/analysis/figures/baseline_comparison'):
    """
    Generate summary table of results
    
    Args:
        baseline_df: Baseline experiment data
        other_df: Other experiment (constrained and adaptive) data
        save_path: Path to save table
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Prepare data
    summary_data = []
    
    # Process baseline data
    for _, row in baseline_df.iterrows():
        summary_data.append({
            'experiment_type': 'Baseline (No Memory)',
            'board_size': str(row['board_size']),
            'agent_a_win_rate': f"{row['agent_a_win_rate']*100:.1f}%",
            'agent_b_win_rate': f"{row['agent_b_win_rate']*100:.1f}%",
            'draw_rate': f"{row['draw_rate']*100:.1f}%",
            'avg_moves': f"{row['avg_moves_per_game']:.1f}",
            'avg_time': f"{row['avg_time_per_game']:.1f}s"
        })
    
    # Process constrained experiment data
    for _, row in other_df.iterrows():
        if row['memory_constraint_a'] in ['graph_only', 'vector_only']:
            experiment_type = f"{row['memory_constraint_a'].replace('_only', ' Memory')}"
            
            # Ensure necessary fields exist
            avg_moves = row.get('avg_moves_per_game', np.nan)
            avg_time = row.get('avg_time_per_game', np.nan)
            
            summary_data.append({
                'experiment_type': experiment_type,
                'board_size': str(row['board_size']),
                'agent_a_win_rate': f"{row['agent_a_win_rate']*100:.1f}%",
                'agent_b_win_rate': f"{row['agent_b_win_rate']*100:.1f}%",
                'draw_rate': f"{row['draw_rate']*100:.1f}%",
                'avg_moves': f"{avg_moves:.1f}" if not np.isnan(avg_moves) else "N/A",
                'avg_time': f"{avg_time:.1f}s" if not np.isnan(avg_time) else "N/A"
            })
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Save as CSV
    summary_df.to_csv(os.path.join(save_path, 'summary_table.csv'), index=False)
    
    # Save as more readable HTML
    html_content = """
    <html>
    <head>
        <style>
            table { border-collapse: collapse; width: 100%; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            tr:hover { background-color: #f5f5f5; }
            h1 { text-align: center; }
        </style>
    </head>
    <body>
        <h1>Memory Experiments Summary</h1>
        """ + summary_df.to_html(index=False) + """
    </body>
    </html>
    """
    
    with open(os.path.join(save_path, 'summary_table.html'), 'w') as f:
        f.write(html_content)
    
    return summary_df

def main():
    """Main function, run all analyses"""
    print("Starting baseline experiment data analysis")
    
    # Load baseline data
    baseline_df = load_baseline_data()
    if baseline_df.empty:
        print("No baseline experiment data found!")
        return
    
    print(f"Loaded {len(baseline_df)} baseline experiment results")
    
    # Load other experiment data
    try:
        other_df = load_other_experiments_data()
        has_other_data = not other_df.empty
        print(f"Loaded {len(other_df)} constrained/adaptive experiment results")
    except Exception as e:
        print(f"Error loading other experiment data: {e}")
        has_other_data = False
        other_df = pd.DataFrame()
    
    # Create output directory
    save_path = 'experiments/results/analysis/figures/baseline_comparison'
    os.makedirs(save_path, exist_ok=True)
    
    # Analyze baseline data
    if has_other_data:
        print("Comparing win rates between baseline and other experiments")
        compare_win_rates(baseline_df, other_df, save_path)
        
        print("Comparing game efficiency between baseline and other experiments")
        compare_game_efficiency(baseline_df, other_df, save_path)
        
        print("Generating summary table")
        generate_summary_table(baseline_df, other_df, save_path)
    else:
        print("Analyzing baseline data only (no other experiment data to compare)")
        # Code for baseline-only analysis could be added here
    
    print(f"Analysis complete, results saved to {save_path}")

if __name__ == "__main__":
    main() 