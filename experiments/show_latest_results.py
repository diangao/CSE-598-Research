#!/usr/bin/env python3
import os
import json
import glob
from pathlib import Path
import argparse
from tabulate import tabulate

def parse_args():
    parser = argparse.ArgumentParser(description='Show latest experiment results')
    parser.add_argument('--board_size', type=str, default=None,
                    help='Filter by board size (e.g., "3x3")')
    parser.add_argument('--run_id', type=int, default=None,
                    help='Filter by run ID')
    parser.add_argument('--num_games', type=int, default=5,
                    help='Number of games to show (default: 5)')
    return parser.parse_args()

def get_latest_stats(board_size=None, run_id=None):
    """Get the latest stats file matching criteria"""
    stats_pattern = "experiments/results/run_*_stats_*.json"
    if board_size and run_id:
        stats_pattern = f"experiments/results/run_{run_id}_stats_{board_size}_*.json"
    elif board_size:
        stats_pattern = f"experiments/results/run_*_stats_{board_size}_*.json"
    elif run_id:
        stats_pattern = f"experiments/results/run_{run_id}_stats_*.json"
    
    stats_files = sorted(glob.glob(stats_pattern), key=os.path.getmtime, reverse=True)
    
    if not stats_files:
        print(f"No stats files found matching criteria")
        return None
    
    latest_stats_file = stats_files[0]
    with open(latest_stats_file, 'r') as f:
        stats = json.load(f)
    
    print(f"Latest stats file: {os.path.basename(latest_stats_file)}")
    return stats

def get_latest_games(num_games=5, board_size=None, run_id=None):
    """Get the latest game logs matching criteria"""
    logs_pattern = "experiments/results/game_logs/run_*_game_*.json"
    if board_size and run_id:
        logs_pattern = f"experiments/results/game_logs/run_{run_id}_game_*_{board_size}_*.json"
    elif board_size:
        logs_pattern = f"experiments/results/game_logs/run_*_game_*_{board_size}_*.json"
    elif run_id:
        logs_pattern = f"experiments/results/game_logs/run_{run_id}_game_*.json"
    
    game_files = sorted(glob.glob(logs_pattern), key=os.path.getmtime, reverse=True)
    
    if not game_files:
        print(f"No game logs found matching criteria")
        return []
    
    games = []
    for i, game_file in enumerate(game_files[:num_games]):
        with open(game_file, 'r') as f:
            game = json.load(f)
        games.append(game)
        
    return games

def print_stats_summary(stats):
    """Print a summary of the statistics"""
    if not stats:
        return
    
    print("\n=== Experiment Statistics ===")
    print(f"Board Size: {stats.get('board_size', 'Unknown')}")
    print(f"Run ID: {stats.get('run_id', 'Unknown')}")
    
    # Create a table for results
    results_table = [
        ["Agent A Wins", stats.get('agent_a_wins', 0)],
        ["Agent B Wins", stats.get('agent_b_wins', 0)],
        ["Draws", stats.get('draws', 0)],
        ["Agent A Total Tokens", stats.get('agent_a_total_tokens', 0)],
        ["Agent B Total Tokens", stats.get('agent_b_total_tokens', 0)]
    ]
    print("\n" + tabulate(results_table, headers=["Metric", "Value"]))
    
    # Create tables for memory usage
    agent_a_memory = stats.get('agent_a_memory_calls', {})
    agent_b_memory = stats.get('agent_b_memory_calls', {})
    
    memory_table = []
    for func in agent_a_memory.keys():
        memory_table.append([
            func, 
            agent_a_memory.get(func, 0),
            agent_b_memory.get(func, 0)
        ])
    
    print("\n=== Memory Function Usage ===")
    print(tabulate(memory_table, headers=["Function", "Agent A", "Agent B"]))

def print_game_summary(games):
    """Print a summary of the games"""
    if not games:
        return
    
    print(f"\n=== Latest {len(games)} Games ===")
    
    game_table = []
    for game in games:
        run_id = game.get('run_id', 'Unknown')
        game_id = game.get('game_id', 'Unknown')
        board_size = game.get('board_size', 'Unknown')
        winner = game.get('winner', 'Draw')
        agent_a_tokens = game.get('agent_a', {}).get('total_tokens_used', 0)
        agent_b_tokens = game.get('agent_b', {}).get('total_tokens_used', 0)
        
        # Get memory usage summaries
        agent_a_memory = game.get('agent_a', {}).get('memory_usage_summary', {})
        agent_b_memory = game.get('agent_b', {}).get('memory_usage_summary', {})
        
        # Calculate total memory calls per agent
        agent_a_mem_count = sum(agent_a_memory.values())
        agent_b_mem_count = sum(agent_b_memory.values())
        
        game_table.append([
            f"{run_id}:{game_id}", 
            board_size, 
            winner, 
            agent_a_tokens,
            agent_b_tokens,
            agent_a_mem_count,
            agent_b_mem_count
        ])
    
    print(tabulate(game_table, headers=[
        "Run:Game ID", "Board Size", "Winner", 
        "A Tokens", "B Tokens", "A Mem Calls", "B Mem Calls"
    ]))

def main():
    args = parse_args()
    
    # Get latest stats and games
    stats = get_latest_stats(args.board_size, args.run_id)
    games = get_latest_games(args.num_games, args.board_size, args.run_id)
    
    # Print summaries
    print_stats_summary(stats)
    print_game_summary(games)

if __name__ == "__main__":
    main() 