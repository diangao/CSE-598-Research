import os
import json
import re
import pandas as pd
import numpy as np
from datetime import datetime
import glob

def get_all_experiment_files():
    """Get paths to all experiment summary files"""
    files = []
    
    # Get experiment files from dian's directory (constrained memory experiments)
    dian_path = "experiments/results/from_dian/"
    if os.path.exists(dian_path):
        for file in os.listdir(dian_path):
            if file.endswith(".json") and file.startswith("exp_memA"):
                files.append(os.path.join(dian_path, file))
    
    # Get experiment files from anish's directory (adaptive memory experiments)
    anish_path = "experiments/results/from_anish/"
    if os.path.exists(anish_path):
        for file in os.listdir(anish_path):
            if file.endswith(".json") and file.startswith("exp_memA"):
                files.append(os.path.join(anish_path, file))
    
    return files

def get_all_game_log_files():
    """Get paths to all game log files"""
    logs = []
    
    # Get game logs from dian's directory (constrained memory experiments)
    dian_logs = "experiments/results/from_dian/game_logs/"
    if os.path.exists(dian_logs):
        for file in os.listdir(dian_logs):
            if file.endswith(".json") and file.startswith("game_run"):
                logs.append(os.path.join(dian_logs, file))
    
    # Get game logs from anish's directory (adaptive memory experiments)
    anish_logs = "experiments/results/from_anish/game_logs/"
    if os.path.exists(anish_logs):
        for file in os.listdir(anish_logs):
            if file.endswith(".json") and file.startswith("game_run"):
                logs.append(os.path.join(anish_logs, file))
    
    return logs

def parse_experiment_filename(filename):
    """Parse experiment summary filename to extract parameters"""
    # Extract base filename without path
    base_name = os.path.basename(filename)
    
    # Set default values
    params = {
        'source': 'unknown',
        'memory_constraint_a': 'unknown',
        'memory_constraint_b': 'unknown',
        'board_size': 'unknown',
        'model': 'unknown',
        'games_count': 0,
        'run_id': 'unknown',
        'experiment_type': 'unknown'
    }
    
    # Check source
    if "from_anish" in filename:
        params['source'] = 'anish'
        params['experiment_type'] = 'adaptive'
    elif "from_dian" in filename:
        params['source'] = 'dian'
        params['experiment_type'] = 'constrained'
    
    # Parse filename components
    if "memA_" in base_name and "memB_" in base_name:
        # Extract memory_constraint_a
        match_a = re.search(r'memA_([a-z_]+)_memB', base_name)
        if match_a:
            params['memory_constraint_a'] = match_a.group(1)
        
        # Extract memory_constraint_b
        match_b = re.search(r'memB_([a-z_]+)_', base_name)
        if match_b:
            params['memory_constraint_b'] = match_b.group(1)
        
        # Extract board_size
        match_size = re.search(r'_(\d+)x\1_', base_name)
        if match_size:
            params['board_size'] = match_size.group(1)
        
        # Extract model
        match_model = re.search(r'_(gpt[^_]+)_', base_name)
        if match_model:
            params['model'] = match_model.group(1)
        
        # Extract games_count
        match_games = re.search(r'_(\d+)games_', base_name)
        if match_games:
            params['games_count'] = int(match_games.group(1))
        
        # Extract run_id
        match_run = re.search(r'_(\d{10})\.json$', base_name)
        if match_run:
            params['run_id'] = match_run.group(1)
    
    return params

def parse_game_log_filename(filename):
    """Parse game log filename to extract parameters"""
    # Extract base filename without path
    base_name = os.path.basename(filename)
    
    # Set default values
    params = {
        'source': 'unknown',
        'run_id': 'unknown',
        'game_id': 0,
        'board_size': 'unknown',
        'memory_constraint_a': 'unknown',
        'memory_constraint_b': 'unknown',
        'model': 'unknown',
        'timestamp': 'unknown',
        'experiment_type': 'unknown'
    }
    
    # Check source
    if "from_anish" in filename:
        params['source'] = 'anish'
        params['experiment_type'] = 'adaptive'
    elif "from_dian" in filename:
        params['source'] = 'dian'
        params['experiment_type'] = 'constrained'
    
    # Parse filename components
    if base_name.startswith("game_run"):
        # Extract run_id
        match_run = re.search(r'game_run(\d+)_', base_name)
        if match_run:
            params['run_id'] = match_run.group(1)
        
        # Extract game_id
        match_id = re.search(r'_id(\d+)_', base_name)
        if match_id:
            params['game_id'] = int(match_id.group(1))
        
        # Extract board_size
        match_size = re.search(r'_size(\d+)_', base_name)
        if match_size:
            params['board_size'] = match_size.group(1)
        
        # Extract memory constraints
        match_mem = re.search(r'_memA([a-z_]+)_memB([a-z_]+)_', base_name)
        if match_mem:
            params['memory_constraint_a'] = match_mem.group(1)
            params['memory_constraint_b'] = match_mem.group(2)
        
        # Extract model
        match_model = re.search(r'_(gpt[^_]+)_', base_name)
        if match_model:
            params['model'] = match_model.group(1)
        
        # Extract timestamp
        match_time = re.search(r'_(\d{8}_\d{6})\.json$', base_name)
        if match_time:
            params['timestamp'] = match_time.group(1)
    
    return params

def load_experiment_data(results_dir=None, experiment_type=None):
    """
    Load all experiment data from JSON files
    
    Args:
        results_dir: Directory containing experiment results (None for both directories)
        experiment_type: Type of experiment to load ('constrained', 'adaptive', or None for both)
        
    Returns:
        DataFrame containing experiment data
    """
    # Determine which directories to search
    search_dirs = []
    if results_dir:
        search_dirs.append(results_dir)
    else:
        if experiment_type == 'constrained' or experiment_type is None:
            search_dirs.append("experiments/results/from_dian")
        if experiment_type == 'adaptive' or experiment_type is None:
            search_dirs.append("experiments/results/from_anish")
    
    # Find all experiment JSON files across specified directories
    json_files = []
    for directory in search_dirs:
        if os.path.exists(directory):
            json_files.extend(glob.glob(f"{directory}/exp_*.json"))
    
    if not json_files:
        print(f"No experiment files found in specified directories")
        return pd.DataFrame()
    
    data = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                exp_data = json.load(f)
                
            # Extract experiment parameters from filename
            filename = os.path.basename(json_file)
            match = re.search(r'exp_memA_(.+)_memB_(.+)_(\d+)x\d+_(.+)_(\d+)games_(\d+)\.json', filename)
            
            if match:
                memory_constraint_a = match.group(1)
                memory_constraint_b = match.group(2)
                board_size = int(match.group(3))
                model = match.group(4).replace('_', '-')
                num_games = int(match.group(5))
                run_id = match.group(6)
                
                # Determine experiment source and type
                source = 'dian' if 'from_dian' in json_file else 'anish'
                exp_type = 'constrained' if source == 'dian' else 'adaptive'
                
                # Add to data list
                exp_data.update({
                    'memory_constraint_a': memory_constraint_a,
                    'memory_constraint_b': memory_constraint_b,
                    'board_size': board_size,
                    'model': model,
                    'num_games': num_games,
                    'run_id': run_id,
                    'filename': filename,
                    'source': source,
                    'experiment_type': exp_type
                })
                
                data.append(exp_data)
            else:
                print(f"Couldn't parse filename: {filename}")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    if not data:
        print("No valid experiment data found")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Calculate additional metrics
    df['agent_a_win_rate'] = df['agent_a_wins'] / df['total_games']
    df['agent_b_win_rate'] = df['agent_b_wins'] / df['total_games'] 
    df['draw_rate'] = df['draws'] / df['total_games']
    
    # Calculate memory usage statistics if available
    if 'agent_a_graph_calls' in df.columns:
        df['agent_a_total_memory_calls'] = df['agent_a_graph_calls'] + df['agent_a_vector_calls'] + df['agent_a_semantic_calls']
        df['agent_b_total_memory_calls'] = df['agent_b_graph_calls'] + df['agent_b_vector_calls'] + df['agent_b_semantic_calls']
    
    # Calculate win-token efficiency if token data available
    if 'agent_a_total_tokens' in df.columns and 'agent_b_total_tokens' in df.columns:
        # Avoid division by zero
        df['agent_a_tokens_per_game'] = df['agent_a_total_tokens'] / df['total_games']
        df['agent_b_tokens_per_game'] = df['agent_b_total_tokens'] / df['total_games']
        
        # Calculate win-token ratio - higher is better (more wins per token)
        df['agent_a_win_token_ratio'] = df['agent_a_win_rate'] / (df['agent_a_tokens_per_game'] / 10000)
        df['agent_b_win_token_ratio'] = df['agent_b_win_rate'] / (df['agent_b_tokens_per_game'] / 10000)
    
    print(f"Loaded {len(df)} experiment results")
    return df

def process_game_logs(logs_dir=None, experiment_type=None):
    """
    Process detailed game logs to extract memory usage patterns
    
    Args:
        logs_dir: Directory containing game log files (None for both directories)
        experiment_type: Type of experiment to process ('constrained', 'adaptive', or None for both)
        
    Returns:
        DataFrame with per-memory-operation data
    """
    # Determine which directories to search
    search_dirs = []
    if logs_dir:
        search_dirs.append(logs_dir)
    else:
        if experiment_type == 'constrained' or experiment_type is None:
            search_dirs.append("experiments/results/from_dian/game_logs")
        if experiment_type == 'adaptive' or experiment_type is None:
            search_dirs.append("experiments/results/from_anish/game_logs")
    
    # Find all game log JSON files across specified directories
    json_files = []
    for directory in search_dirs:
        if os.path.exists(directory):
            json_files.extend(glob.glob(f"{directory}/game_run*.json"))
    
    if not json_files:
        print(f"No game log files found in specified directories")
        return pd.DataFrame()
    
    memory_usage_data = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                game_data = json.load(f)
            
            # Extract run ID and game info from filename
            filename = os.path.basename(json_file)
            match = re.search(r'game_run(\d+)_id(\d+)_size(\d+)_memA(.+)_memB(.+)_(.+)_(\d+)\.json', filename)
            
            if not match:
                continue
                
            run_id = match.group(1)
            game_id = match.group(2)
            board_size = int(match.group(3))
            memory_constraint_a = match.group(4)
            memory_constraint_b = match.group(5)
            model = match.group(6).replace('_', '-')
            
            # Determine experiment source and type
            source = 'dian' if 'from_dian' in json_file else 'anish'
            exp_type = 'constrained' if source == 'dian' else 'adaptive'
            
            # Process memory operations
            if 'turns' in game_data:
                for turn in game_data['turns']:
                    if 'memory_function' in turn and turn['memory_function']:
                        # Parse memory function name to get type and operation
                        mem_func = turn['memory_function']
                        
                        # Skip if not a memory function
                        if not any(x in mem_func for x in ['graph', 'vector', 'semantic']):
                            continue
                            
                        parts = mem_func.split('_')
                        if len(parts) >= 2:
                            memory_type = parts[0].capitalize()
                            operation = parts[1]
                            
                            # Determine game phase based on turn number
                            # This is an approximation - could be made more sophisticated
                            total_turns = len(game_data['turns'])
                            turn_num = turn.get('turn', 0)
                            
                            if turn_num <= total_turns * 0.3:
                                phase = 'early'
                            elif turn_num <= total_turns * 0.7:
                                phase = 'mid'
                            else:
                                phase = 'late'
                            
                            # Determine the agent who made this memory call
                            agent_id = turn.get('agent', 'unknown')
                            
                            # Check if this agent won the game
                            agent_won = False
                            if 'final_result' in game_data:
                                if agent_id == 'agent_a' and game_data['final_result'] == 'agent_a_win':
                                    agent_won = True
                                elif agent_id == 'agent_b' and game_data['final_result'] == 'agent_b_win':
                                    agent_won = True
                            
                            memory_usage_data.append({
                                'run_id': run_id,
                                'game_id': game_id,
                                'board_size': board_size,
                                'memory_constraint_a': memory_constraint_a,
                                'memory_constraint_b': memory_constraint_b,
                                'model': model,
                                'agent_id': agent_id,
                                'turn': turn_num,
                                'memory_type': memory_type,
                                'operation': operation,
                                'phase': phase,
                                'agent_won': agent_won,
                                'tokens_used': turn.get('tokens_used', 0),
                                'source': source,
                                'experiment_type': exp_type
                            })
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    if not memory_usage_data:
        print("No memory usage data found in game logs")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(memory_usage_data)
    print(f"Processed {len(df)} memory operations from game logs")
    return df

if __name__ == "__main__":
    # Test data loading
    print("Loading constrained experiment data (Dian):")
    df_constrained = load_experiment_data(experiment_type='constrained')
    print(df_constrained[['board_size', 'memory_constraint_a', 'memory_constraint_b', 'agent_a_wins', 'agent_b_wins']].head())
    
    print("\nLoading adaptive experiment data (Anish):")
    df_adaptive = load_experiment_data(experiment_type='adaptive')
    if not df_adaptive.empty:
        print(df_adaptive[['board_size', 'memory_constraint_a', 'memory_constraint_b', 'agent_a_wins', 'agent_b_wins']].head())
    else:
        print("No adaptive experiment data found")
    
    # Test game log processing
    print("\nProcessing game logs:")
    memory_df = process_game_logs()
    if not memory_df.empty:
        print(memory_df[['experiment_type', 'board_size', 'memory_type', 'operation']].head()) 