#!/usr/bin/env python3
import os
import subprocess
import logging
import argparse
import json
import time
import shutil
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run memory comparison baseline experiments")
    
    # Experiment parameters
    parser.add_argument('--games', type=int, default=15, 
                        help='Number of games per condition (default: 15)')
    parser.add_argument('--results-dir', type=str, default='experiments/results/memory_baseline',
                        help='Directory to save all results')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='LLM model to use')
    
    # Optional parameters
    parser.add_argument('--skip-3x3', action='store_true',
                        help='Skip 3x3 board experiments')
    parser.add_argument('--skip-9x9', action='store_true',
                        help='Skip 9x9 board experiments')
    parser.add_argument('--skip-graph', action='store_true',
                        help='Skip GraphMemory experiments')
    parser.add_argument('--skip-vector', action='store_true',
                        help='Skip VectorMemory experiments')
    parser.add_argument('--skip-no-memory', action='store_true',
                        help='Skip No-Memory baseline experiments')
    
    return parser.parse_args()

def run_memory_baselines(args):
    """Run memory comparison baseline experiments"""
    
    # Record start time
    start_time = datetime.now()
    logger.info(f"Starting memory comparison baseline experiments at {start_time}")
    
    # 获取绝对路径以确保一致性
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Ensure results directory exists - 使用绝对路径
    results_dir = os.path.join(project_root, args.results_dir.replace('experiments/', ''))
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 确保game_logs目录存在
    game_logs_dir = results_dir / "game_logs"
    game_logs_dir.mkdir(parents=True, exist_ok=True)
    
    # 清理run_self_play.py可能生成的临时结果文件，防止和旧文件混淆
    temp_result_files = list(Path(os.path.join(project_root, "experiments/results")).glob("exp_*.json"))
    if temp_result_files:
        logger.info(f"Removing {len(temp_result_files)} temporary result files from previous runs")
        for file in temp_result_files:
            try:
                os.remove(file)
            except Exception as e:
                logger.warning(f"Could not remove {file}: {e}")
    
    # Define experiment matrix
    board_sizes = []
    if not args.skip_3x3:
        board_sizes.append(3)
    if not args.skip_9x9:
        board_sizes.append(9)
    
    # Define memory constraint matrix
    memory_constraints = []
    if not args.skip_graph:
        memory_constraints.append("graph_only")
    if not args.skip_vector:
        memory_constraints.append("vector_only")
    if not args.skip_no_memory:
        memory_constraints.append("baseline")  # No memory baseline

    # Define agent combinations
    agent_combinations = [
        ("agent_a", "agent_a"),  # Agent A (Memory) vs Agent A (No Memory)
        ("agent_b", "agent_b"),  # Agent B (Memory) vs Agent B (No Memory)
        ("agent_a", "agent_b")   # Agent A (No Memory) vs Agent B (No Memory)
    ]

    # Create an experiment tracker
    experiment_results = []
    
    # Run experiments for each combination
    for board_size in board_sizes:
        for memory_type in memory_constraints:
            if memory_type == "baseline":
                # No memory baseline experiments - run all agent combinations
                for agent_a_type, agent_b_type in agent_combinations:
                    if agent_a_type == agent_b_type:
                        # Skip running agent A vs agent A or agent B vs agent B for baseline
                        # as they would be redundant since both agents have no memory
                        continue
                        
                    logger.info(f"=== Running baseline (no memory) experiment on {board_size}x{board_size} board: {agent_a_type} vs {agent_b_type} ===")
                    result = run_single_baseline_experiment(
                        board_size=board_size,
                        memory_type="baseline",
                        agent_a_type=agent_a_type,
                        agent_b_type=agent_b_type,
                        games=args.games,
                        model=args.model,
                        results_dir=results_dir,
                        project_root=project_root
                    )
                    if result:
                        experiment_results.append({
                            "board_size": board_size,
                            "memory_type": "baseline",
                            "agent_a": f"{agent_a_type} (no memory)",
                            "agent_b": f"{agent_b_type} (no memory)",
                            "status": "success"
                        })
            else:
                # Memory comparison experiments
                for agent_a_type, agent_b_type in agent_combinations:
                    # Skip duplicate experiments
                    if agent_a_type == agent_b_type and memory_type != "baseline":
                        logger.info(f"=== Running {agent_a_type} ({memory_type}) vs {agent_b_type} (no memory) on {board_size}x{board_size} board ===")
                        result = run_single_comparison_experiment(
                            board_size=board_size,
                            memory_type=memory_type,
                            memory_agent_type=agent_a_type,
                            baseline_agent_type=agent_b_type,
                            games=args.games,
                            model=args.model,
                            results_dir=results_dir,
                            project_root=project_root
                        )
                        if result:
                            experiment_results.append({
                                "board_size": board_size,
                                "memory_type": memory_type,
                                "agent_a": f"{agent_a_type} ({memory_type})",
                                "agent_b": f"{agent_b_type} (no memory)",
                                "status": "success"
                            })
    
    # Record end time and create summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60  # minutes
    
    # Create summary report
    summary_path = results_dir / f"memory_baseline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_minutes": duration,
        "experiments": experiment_results,
        "model": args.model,
        "games_per_condition": args.games,
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"=== All memory baseline experiments completed ===")
    logger.info(f"Started: {start_time}")
    logger.info(f"Ended: {end_time}")
    logger.info(f"Duration: {duration:.1f} minutes")
    logger.info(f"Summary written to {summary_path}")
    
    return summary

def run_single_baseline_experiment(board_size, memory_type, agent_a_type, agent_b_type, games, model, results_dir, project_root):
    """Run a single baseline experiment (both agents without memory)"""
    
    # Build experiment ID
    exp_id = f"baseline_nomem_{agent_a_type}vs{agent_b_type}_b{board_size}x{board_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Running baseline experiment: {exp_id}")
    
    # Create the experiment directory
    exp_dir = results_dir / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 确保memory_baseline/game_logs目录存在
    game_logs_dir = results_dir / "game_logs"
    game_logs_dir.mkdir(exist_ok=True, parents=True)
    
    # Use existing run_baseline.py script from project root
    # 使用绝对路径构建命令
    cmd = [
        "python", "-m", "experiments.run_baseline",
        "--board-size", str(board_size),
        "--games", str(games),
        "--results-dir", str(exp_dir),
        "--model", model,
        "--system-a", f"experiments/prompts/system_{agent_a_type}.txt",
        "--system-b", f"experiments/prompts/system_{agent_b_type}.txt"
    ]
    
    # Run command
    logger.info(f"Running command: {' '.join(cmd)}")
    # 在project_root目录下运行命令，确保路径一致性
    process = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
    
    # Check results
    if process.returncode == 0:
        logger.info(f"Baseline experiment {exp_id} completed successfully")
        
        # 现在我们需要从run_baseline.py生成的stats.json中创建一个与其他实验格式兼容的文件
        try:
            # 读取stats.json文件
            stats_path = exp_dir / "stats.json"
            config_path = exp_dir / "config.json"
            
            if stats_path.exists() and config_path.exists():
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # 创建与memory_comparison实验兼容的统计格式
                run_id = int(time.time())
                compatible_stats = {
                    'board_size': f"{board_size}x{board_size}",
                    'agent_a_wins_as_first': stats.get('wins_a_first', 0),
                    'agent_a_wins_as_second': stats.get('wins_a_second', 0),
                    'agent_b_wins_as_first': stats.get('wins_b_first', 0),
                    'agent_b_wins_as_second': stats.get('wins_b_second', 0),
                    'draws': stats.get('draws', 0),
                    'agent_a_total_tokens': stats.get('tokens_a', 0),
                    'agent_b_total_tokens': stats.get('tokens_b', 0),
                    'agent_a_total_wins': stats.get('wins_a', 0),
                    'agent_b_total_wins': stats.get('wins_b', 0),
                    'memory_constraint_a': 'baseline',
                    'memory_constraint_b': 'baseline',
                    'experiment_type': 'memory_comparison',
                    'run_id': run_id,
                    'agent_a_memory_calls': {k: 0 for k in [
                        'graph_store', 'graph_read', 
                        'vector_store', 'vector_read', 
                        'semantic_store', 'semantic_read',
                        'update_graph_schema', 'update_vector_schema', 'update_semantic_schema'
                    ]},
                    'agent_b_memory_calls': {k: 0 for k in [
                        'graph_store', 'graph_read', 
                        'vector_store', 'vector_read', 
                        'semantic_store', 'semantic_read',
                        'update_graph_schema', 'update_vector_schema', 'update_semantic_schema'
                    ]}
                }
                
                # 保存为兼容格式的文件
                timestamp = int(time.time())
                dest_filename = f"exp_baseline_nomem_{agent_a_type}vs{agent_b_type}_{board_size}x{board_size}_{model.replace('-', '_')}_{games}games_{timestamp}.json"
                
                # 保存到主结果目录(memory_baseline)
                with open(results_dir / dest_filename, 'w') as f:
                    json.dump(compatible_stats, f, indent=2)
                
                # 为每个游戏生成一个伪游戏日志，以保持与其他实验一致
                for game_num in range(1, games + 1):
                    game_log = {
                        "run_id": run_id,
                        "game_id": game_num,
                        "timestamp": datetime.now().isoformat(),
                        "board_size": board_size,
                        "agent_a": {
                            "objective": "maximize win rate while minimizing token use",
                            "total_tokens_used": 0,
                            "memory_usage_summary": {k: 0 for k in [
                                'graph_store', 'graph_read', 
                                'vector_store', 'vector_read', 
                                'semantic_store', 'semantic_read',
                                'update_graph_schema', 'update_vector_schema', 'update_semantic_schema'
                            ]},
                            "schema_updates": [],
                            "turns": []
                        },
                        "agent_b": {
                            "objective": "maximize win rate while minimizing token use",
                            "total_tokens_used": 0,
                            "memory_usage_summary": {k: 0 for k in [
                                'graph_store', 'graph_read', 
                                'vector_store', 'vector_read', 
                                'semantic_store', 'semantic_read',
                                'update_graph_schema', 'update_vector_schema', 'update_semantic_schema'
                            ]},
                            "schema_updates": [],
                            "turns": []
                        },
                        "final_result": "baseline_log",
                        "winner": "unknown"
                    }
                    
                    # 使用与GameLogger.end_game()相同的命名格式
                    model_name = model.replace("-", "_").replace("/", "_")
                    game_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    log_filename = f"game_run{run_id}_id{game_num}_size{board_size}_memAbaseline_memBbaseline_{model_name}_{game_timestamp}.json"
                    
                    with open(game_logs_dir / log_filename, 'w') as f:
                        json.dump(game_log, f, indent=2)
                
                logger.info(f"Created compatible stats file: {dest_filename} and game logs in {game_logs_dir}")
            else:
                logger.warning(f"Stats or config file not found in {exp_dir}")
        except Exception as e:
            logger.warning(f"Failed to create compatible stats file: {e}")
        
        return True
    else:
        logger.error(f"Baseline experiment {exp_id} failed")
        logger.error(f"Error: {process.stderr}")
        return False

def run_single_comparison_experiment(board_size, memory_type, memory_agent_type, baseline_agent_type, games, model, results_dir, project_root):
    """Run a memory comparison experiment (one agent with memory, one without)"""
    
    # Build experiment ID
    exp_id = f"compare_{memory_agent_type}_{memory_type}_vs_{baseline_agent_type}_nomem_b{board_size}x{board_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Running comparison experiment: {exp_id}")
    
    # Create results directory for metadata
    exp_dir = results_dir / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 确保统一的game_logs目录存在 - 所有游戏日志都将存储在这里
    game_logs_dir = results_dir / "game_logs"
    game_logs_dir.mkdir(exist_ok=True, parents=True)
    
    # 创建临时游戏日志目录用于运行时
    temp_logs_dir = exp_dir / "temp_game_logs"
    temp_logs_dir.mkdir(exist_ok=True)
    
    # Use run_self_play.py script from project root with appropriate configuration 
    # to ensure we get the same log format
    cmd = [
        "python", "-m", "experiments.run_self_play",
        "--board_size", str(board_size),
        "--num_games", str(games),
        "--model", model,
        "--agent_mode", "submit_board",  # Use simplified board submission mode
        "--use_pretrained_autoencoder",  # Use pretrained autoencoder
        "--memory_reset", "experiment"   # Reset memory at experiment start
    ]
    
    # Configure memory constraints based on settings
    if memory_agent_type == "agent_a":
        cmd.extend([
            "--memory_constraint_a", memory_type,
            "--memory_constraint_b", "baseline"  # Use baseline for no memory
        ])
    else:  # memory_agent_type == "agent_b"
        cmd.extend([
            "--memory_constraint_a", "baseline",
            "--memory_constraint_b", memory_type
        ])
    
    # 设置环境变量传递到子进程，确保GameLogger使用正确的日志目录
    my_env = os.environ.copy()
    my_env["GAME_LOGS_DIR"] = str(temp_logs_dir)
    
    # Run command
    logger.info(f"Running command: {' '.join(cmd)}")
    # 在project_root目录下运行命令，确保路径一致性
    process = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root, env=my_env)
    
    # Check results
    if process.returncode == 0:
        logger.info(f"Comparison experiment {exp_id} completed successfully")
        
        # Save experiment metadata - 这也是为分析提供额外信息
        metadata = {
            "experiment_type": "memory_comparison",
            "board_size": board_size,
            "memory_type": memory_type,
            "memory_agent_type": memory_agent_type,
            "baseline_agent_type": baseline_agent_type,
            "games": games,
            "model": model,
            "timestamp": datetime.now().isoformat()
        }
        with open(exp_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 搜索run_self_play.py生成的统计文件并将其复制到memory_baseline目录
        try:
            # 查找 run_self_play.py 生成的统计文件 - 使用绝对路径
            result_files = list(Path(os.path.join(project_root, "experiments/results")).glob(f"exp_*_{board_size}x{board_size}_*_{games}games_*.json"))
            if result_files:
                latest_file = max(result_files, key=os.path.getctime)
                
                # 使用命名约定使其匹配memory_comparison类型
                timestamp = int(time.time())
                dest_filename = f"exp_compare_{memory_agent_type}_{memory_type}_vs_{baseline_agent_type}_nomem_{board_size}x{board_size}_{model.replace('-', '_')}_{games}games_{timestamp}.json"
                
                # 将文件复制到主结果目录(memory_baseline)，而不是from_dian
                shutil.copy2(latest_file, results_dir / dest_filename)
                
                # 将临时游戏日志目录中的文件移动到统一的game_logs目录
                for log_file in temp_logs_dir.glob("*.json"):
                    shutil.copy2(log_file, game_logs_dir / log_file.name)
                
                # 删除原始临时文件，防止重复
                try:
                    os.remove(latest_file)
                    logger.info(f"Removed temporary file: {latest_file}")
                except Exception as e:
                    logger.warning(f"Could not remove temporary file {latest_file}: {e}")
                
                logger.info(f"Copied experiment results to {results_dir} and game logs to {game_logs_dir}")
            else:
                logger.warning("No result files found, analysis may not work properly")
        except Exception as e:
            logger.warning(f"Failed to process results file: {e}")
        
        return True
    else:
        logger.error(f"Comparison experiment {exp_id} failed")
        logger.error(f"Error: {process.stderr}")
        return False

if __name__ == "__main__":
    args = parse_args()
    run_memory_baselines(args) 