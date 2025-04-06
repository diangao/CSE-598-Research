#!/usr/bin/env python3
import os
import json
import logging
import argparse
import random
import time
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path("experiments/.env"))

# Get delay settings from environment variables, use default values if not found
TURN_DELAY = float(os.getenv("TURN_DELAY", "1.0"))
GAME_DELAY = float(os.getenv("GAME_DELAY", "5.0"))

# Import our modules
from tictactoe import TicTacToe
from utils.memory_ops import MemoryManager
from agent import TicTacToeAgent
from logger import GameLogger

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run TicTacToe self-play experiments')
    parser.add_argument('--num_games', type=int, default=10,
                        help='Number of games to run (default: 10)')
    parser.add_argument('--openai_api_key', type=str, default=None,
                        help='OpenAI API key (default: use from environment or .env file)')
    parser.add_argument('--first_player', type=str, default='random',
                        choices=['agent_a', 'agent_b', 'random'],
                        help='Which agent goes first (default: random)')
    parser.add_argument('--output_dir', type=str, default='experiments/results',
                        help='Directory to store results (default: experiments/results)')
    parser.add_argument('--memory_reset', type=str, default='none',
                        choices=['none', 'game', 'session'],
                        help='When to reset agent memories (default: none)')
    parser.add_argument('--board_size', type=int, default=3, choices=[3, 4, 5, 6],
                        help='Size of the TicTacToe board (NxN) (default: 3, options: 3, 4, 5, 6)')
    
    return parser.parse_args()

def main():
    """Main function to run the self-play experiment"""
    args = parse_args()
    
    # Set up OpenAI API key
    if args.openai_api_key:
        os.environ['OPENAI_API_KEY'] = args.openai_api_key
    elif 'OPENAI_API_KEY' not in os.environ:
        raise ValueError("OpenAI API key must be provided via --openai_api_key, OPENAI_API_KEY environment variable, or .env file")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize game components with specified board size
    game = TicTacToe(board_size=args.board_size)
    game_logger = GameLogger()
    
    # Initialize memory managers for each agent
    memory_manager_a = MemoryManager('agent_a')
    memory_manager_b = MemoryManager('agent_b')
    
    # Initialize agents
    agent_a = TicTacToeAgent('agent_a', memory_manager_a)
    agent_b = TicTacToeAgent('agent_b', memory_manager_b)
    
    # Track statistics
    stats = {
        'board_size': f"{args.board_size}x{args.board_size}",
        'agent_a_wins': 0,
        'agent_b_wins': 0,
        'draws': 0,
        'agent_a_total_tokens': 0,
        'agent_b_total_tokens': 0,
        'agent_a_memory_calls': {
            'graph_store': 0, 'graph_read': 0,
            'vector_store': 0, 'vector_read': 0,
            'semantic_store': 0, 'semantic_read': 0,
            'update_graph_schema': 0, 'update_vector_schema': 0, 'update_semantic_schema': 0
        },
        'agent_b_memory_calls': {
            'graph_store': 0, 'graph_read': 0,
            'vector_store': 0, 'vector_read': 0,
            'semantic_store': 0, 'semantic_read': 0,
            'update_graph_schema': 0, 'update_vector_schema': 0, 'update_semantic_schema': 0
        }
    }
    
    # Define agent objectives (for logging)
    agent_a_objective = "maximize win rate"
    agent_b_objective = "maximize win rate minus token use"
    
    logger.info(f"Starting experiment with board size {args.board_size}x{args.board_size}")
    
    # Run games
    for game_id in range(1, args.num_games + 1):
        percent_complete = (game_id - 1) / args.num_games * 100
        logger.info(f"Starting game {game_id}/{args.num_games} with board size {args.board_size}x{args.board_size} [{percent_complete:.1f}% complete]")
        
        # Reset game state
        game.reset()
        
        # Reset agent message histories
        agent_a.reset()
        agent_b.reset()
        
        # Reset memories if configured
        if args.memory_reset == 'game':
            memory_manager_a.clear_all_memory()
            memory_manager_b.clear_all_memory()
        
        # Determine first player
        if args.first_player == 'random':
            current_agent = random.choice(['agent_a', 'agent_b'])
        else:
            current_agent = args.first_player
        
        # Start logging this game
        game_logger.start_game(game_id, agent_a_objective, agent_b_objective, args.board_size)
        
        # Game loop
        turn_number = 1
        while not game.is_game_over():
            # Get current board state
            board_state = game.get_state()
            
            # Get move from current agent
            if current_agent == 'agent_a':
                move, move_info = agent_a.get_move(board_state)
            else:
                move, move_info = agent_b.get_move(board_state)
            
            # Add a small delay between turns to reduce API rate limit issues
            time.sleep(TURN_DELAY)
            
            # Track tokens used
            tokens_used = move_info.get('tokens_used', 0)
            if current_agent == 'agent_a':
                stats['agent_a_total_tokens'] += tokens_used
            else:
                stats['agent_b_total_tokens'] += tokens_used
            
            # Track memory function calls
            memory_function = move_info.get('memory_function')
            if memory_function:
                if current_agent == 'agent_a':
                    stats['agent_a_memory_calls'][memory_function] += 1
                else:
                    stats['agent_b_memory_calls'][memory_function] += 1
                    
                # 确保记忆功能被正确记录到日志
                logger.info(f"Recorded memory function call: {current_agent} used {memory_function}")
            
            # Log turn information
            rationale = None
            if move_info.get('raw_response'):
                rationale = game_logger.extract_rationale(str(move_info.get('raw_response')))
            
            game_logger.log_turn(
                turn_number=turn_number,
                agent_id=current_agent,
                board_state=board_state,
                action=move,
                memory_function=move_info.get('memory_function'),
                memory_content=move_info.get('memory_content'),
                memory_query=move_info.get('memory_query'),
                schema_update=move_info.get('schema_update'),
                schema_description=move_info.get('schema_description'),
                rationale_excerpt=rationale,
                tokens_used=tokens_used
            )
            
            # Make the move
            if move and game.is_valid_move(move[0], move[1]):
                game.make_move(move[0], move[1])
                logger.info(f"Game {game_id}, Turn {turn_number}: {current_agent} played {move}")
            else:
                # If invalid move, make a random valid move
                valid_moves = game.get_legal_moves()
                if valid_moves:
                    fallback_move = random.choice(valid_moves)
                    game.make_move(fallback_move[0], fallback_move[1])
                    if move is None:
                        logger.warning(f"Game {game_id}, Turn {turn_number}: {current_agent} returned None move (may be stuck in memory functions), using random move {fallback_move} instead")
                    else:
                        logger.warning(f"Game {game_id}, Turn {turn_number}: {current_agent} made invalid move {move}, using random move {fallback_move} instead")
                else:
                    # No valid moves left
                    logger.info(f"Game {game_id}, Turn {turn_number}: No valid moves left, game ending")
                    break
            
            # Switch player
            current_agent = 'agent_b' if current_agent == 'agent_a' else 'agent_a'
            turn_number += 1
        
        # Game over, determine result
        winner = game.get_winner()
        if winner == 'X':
            # First player won
            if args.first_player == 'agent_a' or (args.first_player == 'random' and current_agent == 'agent_b'):
                result = "win"
                winner_id = "agent_a"
                stats['agent_a_wins'] += 1
            else:
                result = "win"
                winner_id = "agent_b"
                stats['agent_b_wins'] += 1
        elif winner == 'O':
            # Second player won
            if args.first_player == 'agent_b' or (args.first_player == 'random' and current_agent == 'agent_a'):
                result = "win"
                winner_id = "agent_b"
                stats['agent_b_wins'] += 1
            else:
                result = "win"
                winner_id = "agent_a"
                stats['agent_a_wins'] += 1
        else:
            # Draw
            result = "draw"
            winner_id = None
            stats['draws'] += 1
        
        # End game logging
        log_file = game_logger.end_game(result, winner_id)
        logger.info(f"Game {game_id} ended: {result}, winner: {winner_id}. Log saved to {log_file}")
        
        # Longer delay between games to avoid rate limiting
        logger.info(f"Waiting {GAME_DELAY} seconds before starting next game to avoid API rate limits...")
        time.sleep(GAME_DELAY)
    
    # Save overall statistics
    run_id = game_logger.run_id
    stats['run_id'] = run_id
    stats_file = output_dir / f"run_{run_id}_stats_{args.board_size}x{args.board_size}_{int(time.time())}.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    logger.info(f"\n===== Experiment Summary (Board Size: {args.board_size}x{args.board_size}) =====")
    logger.info(f"Total games: {args.num_games}")
    logger.info(f"Agent A wins: {stats['agent_a_wins']} ({stats['agent_a_wins']/args.num_games*100:.1f}%)")
    logger.info(f"Agent B wins: {stats['agent_b_wins']} ({stats['agent_b_wins']/args.num_games*100:.1f}%)")
    logger.info(f"Draws: {stats['draws']} ({stats['draws']/args.num_games*100:.1f}%)")
    logger.info(f"Agent A total tokens: {stats['agent_a_total_tokens']}")
    logger.info(f"Agent B total tokens: {stats['agent_b_total_tokens']}")
    logger.info(f"Statistics saved to {stats_file}")

if __name__ == "__main__":
    main() 