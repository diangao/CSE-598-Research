import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

def load_memory_summaries(directory):
    data = []
    files = sorted(glob(os.path.join(directory, "game_run*.json")))
    for file in files:
        with open(file, 'r') as f:
            game = json.load(f)
        
        board_size = game.get("board_size", "unknown")
        
        for agent_key in ['agent_a', 'agent_b']:
            agent = game.get(agent_key, {})
            memory_usage = agent.get('memory_usage_summary', {})
            
            data.append({
                "agent_id": agent_key,
                "board_size": board_size,
                "graph": memory_usage.get("graph_store", 0) + memory_usage.get("graph_read", 0),
                "vector": memory_usage.get("vector_store", 0) + memory_usage.get("vector_read", 0),
                "semantic": memory_usage.get("semantic_store", 0) + memory_usage.get("semantic_read", 0)
            })
    
    return pd.DataFrame(data)

def compute_memory_type_preference(df):
    usage_totals = df.groupby('agent_id')[['graph', 'vector', 'semantic']].sum()
    usage_props = usage_totals.div(usage_totals.sum(axis=1), axis=0)
    print("\n--- Memory Type Preference (Proportions) ---")
    print(usage_props)
    return usage_props

def plot_memory_type_preference(pref_counts):
    pref_counts.plot(kind='bar', stacked=True)
    plt.title("Memory Type Preference per Agent")
    plt.xlabel("Agent ID")
    plt.ylabel("Proportion")
    plt.legend(title="Memory Type")
    plt.tight_layout()
    plt.savefig("memory_type_preference.png")
    plt.show()

def main():
    directory = "./experiments/results/from_anish/game_logs"
    df = load_memory_summaries(directory)
    
    if df.empty:
        return
    
    pref_counts = compute_memory_type_preference(df)
    plot_memory_type_preference(pref_counts)

if __name__ == "__main__":
    main()
