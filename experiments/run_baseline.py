import os
import argparse
import json
import time
import logging
import random
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

from experiments.utils.memory_ops_baseline import MemoryManager
from experiments.agent import TicTacToeAgent
from experiments.tictactoe import TicTacToe

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义简单的 ResultLogger 类
class ResultLogger:
    """简单的结果记录器，用于保存游戏结果"""
    
    def __init__(self, results_dir):
        """初始化结果记录器"""
        self.results_dir = Path(results_dir)
        self.games_dir = self.results_dir / "games"
        self.games_dir.mkdir(parents=True, exist_ok=True)
        
    def log_game(self, game_data):
        """记录单个游戏的结果"""
        game_id = game_data.get("game_id", 0)
        game_file = self.games_dir / f"game_{game_id}.json"
        
        with open(game_file, 'w') as f:
            json.dump(game_data, f, indent=2)
        
        return True

# 定义 GameResult 类（原来在别处定义但需要在这里使用）
@dataclass
class GameResult:
    """结果数据类，存储游戏结果信息"""
    outcome: str  # "A_WINS", "B_WINS", "DRAW", "TIMEOUT"
    total_moves: int
    final_state: str
    winner: Optional[str] = None  # "A", "B", None

def parse_args():
    parser = argparse.ArgumentParser(description="Run TicTacToe baseline experiments (no memory)")
    
    # 核心参数
    parser.add_argument('--board-size', type=int, default=3, 
                        choices=[3, 6, 9], help='Board size (3x3, 6x6, or 9x9)')
    parser.add_argument('--games', type=int, default=15, 
                        help='Number of games to play (default: 15)')
    parser.add_argument('--results-dir', type=str, default='experiments/results/baseline',
                        help='Directory to save results')
    
    # 系统提示文件
    parser.add_argument('--system-a', type=str, default='experiments/prompts/system_agent_a.txt',
                        help='System prompt file for Agent A')
    parser.add_argument('--system-b', type=str, default='experiments/prompts/system_agent_b.txt',
                        help='System prompt file for Agent B')
    
    # 可选参数
    parser.add_argument('--timeout', type=int, default=60,
                        help='Timeout for agent moves in seconds')
    parser.add_argument('--max-tokens', type=int, default=1024,
                        help='Max tokens for agent responses')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for agent responses')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='LLM model to use')
    
    return parser.parse_args()

def run_baseline_experiment(args):
    """运行 baseline 实验 (无记忆)"""
    
    # 实验信息
    exp_id = f"baseline_b{args.board_size}x{args.board_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Starting baseline experiment: {exp_id}")
    
    # 创建结果目录
    results_dir = Path(args.results_dir) / exp_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 记录实验参数
    with open(results_dir / "config.json", 'w') as f:
        config = vars(args)
        config["exp_id"] = exp_id
        config["timestamp"] = datetime.now().isoformat()
        config["type"] = "baseline_no_memory"
        json.dump(config, f, indent=2)
    
    # 初始化结果记录器
    result_logger = ResultLogger(results_dir)
    
    # 游戏结果统计
    stats = {
        "wins_a": 0,
        "wins_b": 0,
        "draws": 0,
        "timeouts": 0,
        "errors": 0,
        "total_moves": 0,
        "total_time": 0,
        "games": []
    }
    
    # 设置环境变量
    os.environ['OPENAI_MODEL'] = args.model
    
    # 运行多局游戏
    for game_idx in range(args.games):
        logger.info(f"Starting game {game_idx+1}/{args.games}")
        
        # 为每个智能体创建无记忆管理器
        memory_a = MemoryManager(agent_id="agent_a")
        memory_b = MemoryManager(agent_id="agent_b")
        
        # 创建智能体
        agent_a = TicTacToeAgent("agent_a", memory_a)
        agent_b = TicTacToeAgent("agent_b", memory_b)
        
        # 创建游戏
        game = TicTacToe(board_size=args.board_size)
        
        # 运行一局游戏
        start_time = time.time()
        try:
            # 游戏循环
            current_player = 'X'  # X 先走
            total_moves = 0
            
            while not game.is_game_over():
                # 确定当前智能体
                if current_player == 'X':
                    current_agent = agent_a
                    agent_id = "agent_a"
                else:
                    current_agent = agent_b
                    agent_id = "agent_b"
                
                # 获取棋盘状态
                board_state = game.get_state()
                
                # 获取智能体走法
                validator = lambda r, c: game.is_valid_move(r, c)
                move, move_info = current_agent.submit_board(board_state, validator)
                
                # 执行走法
                if move and game.is_valid_move(move[0], move[1]):
                    game.make_move(move[0], move[1])
                    logger.info(f"Game {game_idx+1}, Move {total_moves+1}: {agent_id} played {move}")
                else:
                    # 如果走法无效，随机走一步
                    valid_moves = game.get_legal_moves()
                    if valid_moves:
                        fallback_move = random.choice(valid_moves)
                        game.make_move(fallback_move[0], fallback_move[1])
                        logger.warning(f"Invalid move from {agent_id}, using random move {fallback_move} instead")
                    else:
                        # 没有有效走法
                        logger.info(f"No valid moves left, game ending")
                        break
                
                # 切换玩家
                current_player = 'O' if current_player == 'X' else 'X'
                total_moves += 1
                
                # 检查超时
                if time.time() - start_time > args.timeout:
                    logger.warning(f"Game {game_idx+1} timed out after {total_moves} moves")
                    result = GameResult(
                        outcome="TIMEOUT",
                        total_moves=total_moves,
                        final_state=game.get_state(),
                        winner=None
                    )
                    stats["timeouts"] += 1
                    break
            
            # 游戏结束，确定结果
            game_time = time.time() - start_time
            
            if not game.is_game_over():
                # 超时导致的结束已经在上面处理
                pass
            else:
                winner = game.get_winner()
                if winner == 'X':
                    # X 胜利，agent_a 获胜
                    result = GameResult(
                        outcome="A_WINS",
                        total_moves=total_moves,
                        final_state=game.get_state(),
                        winner="A"
                    )
                    stats["wins_a"] += 1
                    logger.info(f"Game {game_idx+1} result: Agent A wins")
                elif winner == 'O':
                    # O 胜利，agent_b 获胜
                    result = GameResult(
                        outcome="B_WINS",
                        total_moves=total_moves,
                        final_state=game.get_state(),
                        winner="B"
                    )
                    stats["wins_b"] += 1
                    logger.info(f"Game {game_idx+1} result: Agent B wins")
                else:
                    # 平局
                    result = GameResult(
                        outcome="DRAW",
                        total_moves=total_moves,
                        final_state=game.get_state(),
                        winner=None
                    )
                    stats["draws"] += 1
                    logger.info(f"Game {game_idx+1} result: Draw")
            
            # 记录结果
            logger.info(f"Total moves: {total_moves}, time: {game_time:.2f}s")
            
            # 更新统计
            stats["total_moves"] += total_moves
            stats["total_time"] += game_time
            
            # 记录游戏详情
            game_record = {
                "game_id": game_idx,
                "outcome": result.outcome,
                "moves": result.total_moves,
                "time": game_time,
                "board_size": args.board_size,
                "final_state": result.final_state
            }
            stats["games"].append(game_record)
            
            # 保存到日志
            result_logger.log_game(game_record)
            
        except Exception as e:
            logger.error(f"Error in game {game_idx+1}: {e}")
            stats["errors"] += 1
            
        # 重置智能体
        agent_a.reset()
        agent_b.reset()
        
        # 保存当前统计
        with open(results_dir / "stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
    
    # 完成实验，打印最终统计
    if args.games > 0:
        logger.info(f"Experiment completed: {exp_id}")
        logger.info(f"Agent A wins: {stats['wins_a']} ({stats['wins_a']/args.games:.0%})")
        logger.info(f"Agent B wins: {stats['wins_b']} ({stats['wins_b']/args.games:.0%})")
        logger.info(f"Draws: {stats['draws']} ({stats['draws']/args.games:.0%})")
        logger.info(f"Timeouts: {stats['timeouts']} ({stats['timeouts']/args.games:.0%})")
        logger.info(f"Errors: {stats['errors']} ({stats['errors']/args.games:.0%})")
        logger.info(f"Avg moves/game: {stats['total_moves']/args.games:.1f}")
        logger.info(f"Avg time/game: {stats['total_time']/args.games:.1f}s")
    
    return stats

if __name__ == "__main__":
    args = parse_args()
    run_baseline_experiment(args) 