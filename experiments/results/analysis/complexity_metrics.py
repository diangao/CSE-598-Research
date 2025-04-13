import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

def load_json_data(directory):
    data = []
    files = sorted(glob(os.path.join(directory, "*.json")))
    for file in files:
        with open(file, 'r') as f:
            exp = json.load(f)
        
        board_size = int(exp["board_size"].split("x")[0])
        total_games = exp["agent_a_total_wins"] + exp["agent_b_total_wins"] + exp["draws"]
        
        agent_a_memory_calls = sum(exp["agent_a_memory_calls"].values())
        agent_b_memory_calls = sum(exp["agent_b_memory_calls"].values())
        
        data.append({
            "board_size": board_size,
            "agent_a_total_tokens": exp["agent_a_total_tokens"],
            "agent_b_total_tokens": exp["agent_b_total_tokens"],
            "agent_a_memory_calls": agent_a_memory_calls,
            "agent_b_memory_calls": agent_b_memory_calls,
            "agent_a_total_wins": exp["agent_a_total_wins"],
            "agent_b_total_wins": exp["agent_b_total_wins"],
            "draws": exp["draws"],
            "total_games": total_games,
        })
    
    return pd.DataFrame(data)

def compute_growth_rate(x, y, label):
    log_y = np.log1p(y)
    slope, _ = np.polyfit(x, log_y, 1)
    print(f"{label} Growth Rate: {slope:.4f}")
    return slope

def plot_trend(x, y_a, y_b, ylabel, title, filename):
    plt.figure()
    plt.plot(x, y_a, marker='o', label="Agent A")
    plt.plot(x, y_b, marker='s', label="Agent B")
    plt.xlabel("Board Size")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

def main():
    directory = "./experiments/results/from_anish"
    df = load_json_data(directory)
    if df.empty:
        return
    
    board_sizes = df['board_size'].astype(int)
    mem_calls_a = df['agent_a_memory_calls']
    mem_calls_b = df['agent_b_memory_calls']
    tokens_a = df['agent_a_total_tokens'] / df['total_games']
    tokens_b = df['agent_b_total_tokens'] / df['total_games']
    win_rate_a = df['agent_a_total_wins'] / df['total_games']
    win_rate_b = df['agent_b_total_wins'] / df['total_games']
    
    print("\n--- Complexity Adaptation Metrics ---\n")
    compute_growth_rate(board_sizes, mem_calls_a, "Agent A Memory Call")
    compute_growth_rate(board_sizes, mem_calls_b, "Agent B Memory Call")
    compute_growth_rate(board_sizes, tokens_a, "Agent A Token Usage")
    compute_growth_rate(board_sizes, tokens_b, "Agent B Token Usage")
    
    if len(win_rate_a) >= 2 and len(win_rate_b) >= 2:
        retention_a = win_rate_a.iloc[-1] / win_rate_a.iloc[0]
        retention_b = win_rate_b.iloc[-1] / win_rate_b.iloc[0]
        print(f"Agent A Performance Retention Rate: {retention_a:.4f}")
        print(f"Agent B Performance Retention Rate: {retention_b:.4f}")
    
    print("\nPlotting trends...")
    plot_trend(board_sizes, mem_calls_a, mem_calls_b, "Total Memory Calls", "Memory Calls vs Board Size", "memory_calls_vs_board_size.png")
    plot_trend(board_sizes, tokens_a, tokens_b, "Avg Tokens per Game", "Token Usage vs Board Size", "token_usage_vs_board_size.png")
    plot_trend(board_sizes, win_rate_a * 100, win_rate_b * 100, "Win Rate (%)", "Win Rate vs Board Size", "win_rate_vs_board_size.png")

if __name__ == "__main__":
    main()
