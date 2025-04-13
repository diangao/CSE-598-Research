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
    dian_path = "../from_dian/"
    if os.path.exists(dian_path):
        for file in os.listdir(dian_path):
            if file.endswith(".json") and file.startswith("exp_memA"):
                files.append(os.path.join(dian_path, file))
    
    # Get experiment files from anish's directory (adaptive memory experiments)
    anish_path = "../from_anish/"
    if os.path.exists(anish_path):
        for file in os.listdir(anish_path):
            if file.endswith(".json") and file.startswith("exp_memA"):
                files.append(os.path.join(anish_path, file))
    
    return files

def get_all_game_log_files():
    """Get paths to all game log files"""
    logs = []
    
    # Get game logs from dian's directory (constrained memory experiments)
    dian_logs = "../from_dian/game_logs/"
    if os.path.exists(dian_logs):
        for file in os.listdir(dian_logs):
            if file.endswith(".json") and file.startswith("game_run"):
                logs.append(os.path.join(dian_logs, file))
    
    # Get game logs from anish's directory (adaptive memory experiments)
    anish_logs = "../from_anish/game_logs/"
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
            search_dirs.append("../from_dian")
        if experiment_type == 'adaptive' or experiment_type is None:
            search_dirs.append("../from_anish")
    
    # Find all experiment JSON files across specified directories
    json_files = []
    for directory in search_dirs:
        if os.path.exists(directory):
            json_files.extend(glob.glob(f"{directory}/exp_*.json"))
    
    if not json_files:
        print(f"No experiment files found in specified directories: {search_dirs}")
        return pd.DataFrame()
    
    print(f"Found {len(json_files)} experiment files")
    
    data = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                exp_data = json.load(f)
            
            # Get basic info from filename
            filename = os.path.basename(json_file)
            
            # Extract some info from filename if not present in the JSON
            if 'filename' not in exp_data:
                exp_data['filename'] = filename
            
            # Determine experiment source and type
            if 'from_dian' in json_file:
                exp_data['source'] = 'dian'
                exp_data['experiment_type'] = 'constrained'
            elif 'from_anish' in json_file:
                exp_data['source'] = 'anish'
                exp_data['experiment_type'] = 'adaptive'
            else:
                exp_data['source'] = 'unknown'
                exp_data['experiment_type'] = 'unknown'
            
            # Clean up the agent memory call data
            if 'agent_a_memory_calls' in exp_data:
                agent_a_calls = exp_data['agent_a_memory_calls']
                exp_data['agent_a_graph_calls'] = agent_a_calls.get('graph_store', 0) + agent_a_calls.get('graph_read', 0) + agent_a_calls.get('update_graph_schema', 0)
                exp_data['agent_a_vector_calls'] = agent_a_calls.get('vector_store', 0) + agent_a_calls.get('vector_read', 0) + agent_a_calls.get('update_vector_schema', 0)
                exp_data['agent_a_semantic_calls'] = agent_a_calls.get('semantic_store', 0) + agent_a_calls.get('semantic_read', 0) + agent_a_calls.get('update_semantic_schema', 0)
                exp_data['agent_a_total_memory_calls'] = exp_data['agent_a_graph_calls'] + exp_data['agent_a_vector_calls'] + exp_data['agent_a_semantic_calls']
            
            if 'agent_b_memory_calls' in exp_data:
                agent_b_calls = exp_data['agent_b_memory_calls']
                exp_data['agent_b_graph_calls'] = agent_b_calls.get('graph_store', 0) + agent_b_calls.get('graph_read', 0) + agent_b_calls.get('update_graph_schema', 0)
                exp_data['agent_b_vector_calls'] = agent_b_calls.get('vector_store', 0) + agent_b_calls.get('vector_read', 0) + agent_b_calls.get('update_vector_schema', 0)
                exp_data['agent_b_semantic_calls'] = agent_b_calls.get('semantic_store', 0) + agent_b_calls.get('semantic_read', 0) + agent_b_calls.get('update_semantic_schema', 0)
                exp_data['agent_b_total_memory_calls'] = exp_data['agent_b_graph_calls'] + exp_data['agent_b_vector_calls'] + exp_data['agent_b_semantic_calls']
            
            # Compute total games
            total_games = 0
            if 'agent_a_total_wins' in exp_data and 'agent_b_total_wins' in exp_data and 'draws' in exp_data:
                total_games = exp_data['agent_a_total_wins'] + exp_data['agent_b_total_wins'] + exp_data['draws']
                exp_data['total_games'] = total_games
            
            data.append(exp_data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    if not data:
        print("No valid experiment data found")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Calculate additional metrics
    if 'agent_a_total_wins' in df.columns and 'total_games' in df.columns:
        df['agent_a_win_rate'] = df['agent_a_total_wins'] / df['total_games']
    if 'agent_b_total_wins' in df.columns and 'total_games' in df.columns:
        df['agent_b_win_rate'] = df['agent_b_total_wins'] / df['total_games']
    if 'draws' in df.columns and 'total_games' in df.columns:
        df['draw_rate'] = df['draws'] / df['total_games']
    
    # Calculate first-player advantage
    if all(col in df.columns for col in ['agent_a_wins_as_first', 'agent_b_wins_as_first', 'total_games']):
        df['first_player_wins'] = df['agent_a_wins_as_first'] + df['agent_b_wins_as_first'] 
        df['first_player_win_rate'] = df['first_player_wins'] / df['total_games']
        df['first_player_advantage'] = df['first_player_win_rate'] - 0.5
    
    # Calculate tokens per game
    for agent in ['a', 'b']:
        token_col = f'agent_{agent}_total_tokens'
        if token_col in df.columns and 'total_games' in df.columns:
            df[f'agent_{agent}_tokens_per_game'] = df[token_col] / df['total_games']
            
            # Calculate win/token efficiency ratio
            win_col = f'agent_{agent}_win_rate'
            if win_col in df.columns:
                df[f'agent_{agent}_win_token_ratio'] = (df[win_col] * 10000) / df[f'agent_{agent}_tokens_per_game']
    
    # Filter by experiment type if specified
    if experiment_type:
        df = df[df['experiment_type'] == experiment_type]
    
    print(f"Loaded {len(df)} experiment results")
    return df

def process_game_logs(logs_dir=None, experiment_type=None):
    """
    Process all game log files to extract memory usage information
    
    Args:
        logs_dir: Directory containing game logs (None for both directories)
        experiment_type: Type of experiment to load ('constrained', 'adaptive', or None for both)
        
    Returns:
        DataFrame containing memory usage data
    """
    # Determine which directories to search
    search_dirs = []
    if logs_dir:
        search_dirs.append(logs_dir)
    else:
        if experiment_type == 'constrained' or experiment_type is None:
            search_dirs.append("../from_dian/game_logs")
        if experiment_type == 'adaptive' or experiment_type is None:
            search_dirs.append("../from_anish/game_logs")
    
    # Find all game log JSON files across specified directories
    log_files = []
    for directory in search_dirs:
        if os.path.exists(directory):
            log_files.extend(glob.glob(f"{directory}/game_run*.json"))
    
    if not log_files:
        print(f"No game log files found in specified directories")
        return pd.DataFrame()
    
    memory_data = []
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                game_data = json.load(f)
            
            # Extract game parameters from filename
            filename = os.path.basename(log_file)
            file_params = parse_game_log_filename(log_file)
            
            # Skip if not the requested experiment type
            if experiment_type and file_params['experiment_type'] != experiment_type:
                continue
            
            # Process turns for memory usage
            if 'turns' in game_data and isinstance(game_data['turns'], list):
                total_turns = len(game_data['turns'])
                
                for turn_idx, turn in enumerate(game_data['turns']):
                    # Determine game phase
                    if turn_idx < total_turns / 3:
                        phase = 'early'
                    elif turn_idx < 2 * total_turns / 3:
                        phase = 'mid'
                    else:
                        phase = 'late'
                    
                    # Get memory function calls
                    memory_func = turn.get('memory_function', None)
                    if memory_func:
                        # Determine memory type and operation
                        if memory_func.startswith('graph'):
                            memory_type = 'graph'
                        elif memory_func.startswith('vector'):
                            memory_type = 'vector'
                        elif memory_func.startswith('semantic'):
                            memory_type = 'semantic'
                        else:
                            memory_type = 'unknown'
                            
                        if '_store' in memory_func:
                            operation = 'store'
                        elif '_read' in memory_func:
                            operation = 'read'
                        elif 'update_' in memory_func and '_schema' in memory_func:
                            operation = 'schema'
                        else:
                            operation = 'unknown'
                        
                        # Agent info
                        agent_id = turn.get('agent', 'unknown')
                        agent_won = 'final_result' in game_data and game_data['final_result'] == agent_id
                        
                        memory_data.append({
                            'run_id': file_params['run_id'],
                            'game_id': file_params['game_id'],
                            'board_size': file_params['board_size'],
                            'memory_constraint_a': file_params['memory_constraint_a'],
                            'memory_constraint_b': file_params['memory_constraint_b'],
                            'model': file_params['model'],
                            'experiment_type': file_params['experiment_type'],
                            'agent_id': agent_id,
                            'turn': turn_idx,
                            'phase': phase,
                            'memory_function': memory_func,
                            'memory_type': memory_type,
                            'operation': operation,
                            'agent_won': agent_won
                        })
        except Exception as e:
            print(f"Error processing {log_file}: {e}")
    
    if not memory_data:
        print("No memory usage data found in logs")
        return pd.DataFrame()
    
    return pd.DataFrame(memory_data)

if __name__ == "__main__":
    # Test loading experiment data
    print("Loading constrained experiment data (Dian):")
    df_constrained = load_experiment_data(experiment_type='constrained')
    if not df_constrained.empty:
        print(df_constrained[['board_size', 'memory_constraint_a', 'memory_constraint_b', 'agent_a_wins', 'agent_b_wins']].head())
    else:
        print("No constrained experiment data found")
    
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