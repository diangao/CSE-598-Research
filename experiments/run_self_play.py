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
from experiments.tictactoe import TicTacToe
from experiments.utils.memory_ops import MemoryManager
from experiments.agent import TicTacToeAgent
from experiments.logger import GameLogger

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
                        help='Which player goes first (default: random)')
    parser.add_argument('--board_size', type=int, default=3,
                        help='Board size for TicTacToe (default: 3)')
    parser.add_argument('--memory_reset', type=str, default='none',
                        choices=['none', 'game', 'experiment'],
                        help='When to reset agent memories (default: none)')
    parser.add_argument('--turn_delay', type=float, default=None,
                        help='Delay between turns in seconds (default: use from .env)')
    parser.add_argument('--game_delay', type=float, default=None,
                        help='Delay between games in seconds (default: use from .env)')
    parser.add_argument('--model', type=str, default=None,
                        help='OpenAI model to use (default: use from .env)')
    parser.add_argument('--train_autoencoder', action='store_true',
                        help='Train autoencoder on collected board states after each game')
    parser.add_argument('--autoencoder_epochs', type=int, default=200,
                        help='Number of epochs for autoencoder training (default: 200)')
    parser.add_argument('--agent_mode', type=str, choices=['get_move', 'submit_board'], default='get_move',
                        help='Method to use for agent moves: get_move (original) or submit_board (simplified)')
    return parser.parse_args()

def main():
    """Main function to run experiments"""
    # Parse arguments
    args = parse_args()
    
    # Set up OpenAI API key
    if args.openai_api_key:
        os.environ['OPENAI_API_KEY'] = args.openai_api_key
    elif 'OPENAI_API_KEY' not in os.environ:
        raise ValueError("OpenAI API key must be provided via --openai_api_key, OPENAI_API_KEY environment variable, or .env file")
    
    # Create output directory
    output_dir = Path("experiments/results")
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
        'agent_a_wins_as_first': 0,
        'agent_a_wins_as_second': 0,
        'agent_b_wins_as_first': 0,
        'agent_b_wins_as_second': 0,
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
        },
        'agent_a_total_wins': 0,
        'agent_b_total_wins': 0
    }
    
    # Define agent objectives (for logging)
    agent_a_objective = "maximize win rate"
    agent_b_objective = "maximize win rate while minimizing token use"
    
    logger.info(f"Starting experiment with board size {args.board_size}x{args.board_size}")
    
    # Main experiment loop
    for game_num in range(1, args.num_games + 1):
        logger.info(f"Starting game {game_num} of {args.num_games}")
        
        # Reset game
        game.reset()
        
        # Reset agent message history (but not memories unless specified)
        agent_a.reset()
        agent_b.reset()
        
        # Reset memories if specified
        if args.memory_reset == 'game':
            memory_manager_a.clear_all_memory()
            memory_manager_b.clear_all_memory()
            logger.info("Cleared memories for both agents for new game")
        
        # Set agent objectives
        agent_a_objective = "maximize win rate"
        agent_b_objective = "maximize win rate while minimizing token use"
        
        # Determine first player
        if args.first_player == 'random':
            # Randomly select X or O
            first_player_symbol = random.choice(['X', 'O'])
        elif args.first_player == 'agent_a':
            first_player_symbol = 'X'  # agent_a always plays as X when explicitly selected
        else:  # agent_b
            first_player_symbol = 'O'  # agent_b always plays as O when explicitly selected
        
        # Set initial current player
        current_player = first_player_symbol
        
        # Start logging this game
        game_logger.start_game(game_num, agent_a_objective, agent_b_objective, args.board_size)
        
        # Game loop
        turn_number = 0
        while not game.is_game_over():
            turn_number += 1
            
            # Determine which agent's turn it is
            if current_player == 'X':
                current_agent = agent_a
                agent_id = "agent_a"
            else:
                current_agent = agent_b
                agent_id = "agent_b"
            
            # Get the board state
            board_state = game.get_state()
            
            # Get move from agent based on the selected mode
            if args.agent_mode == 'get_move':
                # Use original get_move method
                move, move_info = current_agent.get_move(board_state)
                game_logger.log_move(agent_id, move, move_info)
            else:
                # Use simplified submit_board method
                validator = lambda r, c: game.is_valid_move(r, c)
                move, move_info = current_agent.submit_board(board_state, validator)
                game_logger.log_move(agent_id, move, move_info)
            
            # Track tokens used
            tokens_used = move_info.get('tokens_used', 0)
            if agent_id == 'agent_a':
                stats['agent_a_total_tokens'] += tokens_used
            else:
                stats['agent_b_total_tokens'] += tokens_used
            
            # Track memory function calls
            memory_function = move_info.get('memory_function')
            if memory_function:
                if agent_id == 'agent_a':
                    stats['agent_a_memory_calls'][memory_function] += 1
                else:
                    stats['agent_b_memory_calls'][memory_function] += 1
                    
                # 确保记忆功能被正确记录到日志
                logger.info(f"Recorded memory function call: {agent_id} used {memory_function}")
            
            # Log turn information
            rationale = None
            if move_info.get('raw_response'):
                rationale = game_logger.extract_rationale(str(move_info.get('raw_response')))
            
            game_logger.log_turn(
                turn_number=turn_number,
                agent_id=agent_id,
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
            
            # Make the move if valid, otherwise use fallback
            if move and game.is_valid_move(move[0], move[1]):
                game.make_move(move[0], move[1])
                logger.info(f"Game {game_num}, Turn {turn_number}: {agent_id} played {move}")
            else:
                # If invalid move, make a random valid move
                valid_moves = game.get_legal_moves()
                if valid_moves:
                    fallback_move = random.choice(valid_moves)
                    game.make_move(fallback_move[0], fallback_move[1])
                    if move is None:
                        logger.warning(f"Game {game_num}, Turn {turn_number}: {agent_id} returned None move (may be stuck in memory functions), using random move {fallback_move} instead")
                    else:
                        logger.warning(f"Game {game_num}, Turn {turn_number}: {agent_id} made invalid move {move}, using random move {fallback_move} instead")
                else:
                    # No valid moves left
                    logger.info(f"Game {game_num}, Turn {turn_number}: No valid moves left, game ending")
                    break
            
            # Switch player
            current_player = 'O' if current_player == 'X' else 'X'
        
        # Game over, determine result
        winner = game.get_winner()
        if winner == 'X':
            # X won
            winner_id = "agent_a"  # Since X is always agent_a
            if first_player_symbol == 'X':
                # Agent A went first and won
                result = "win"
                stats['agent_a_wins_as_first'] += 1
            else:
                # Agent A went second and won
                result = "win"
                stats['agent_a_wins_as_second'] += 1
            stats['agent_a_total_wins'] += 1
        elif winner == 'O':
            # O won
            winner_id = "agent_b"  # Since O is always agent_b
            if first_player_symbol == 'O':
                # Agent B went first and won
                result = "win"
                stats['agent_b_wins_as_first'] += 1
            else:
                # Agent B went second and won
                result = "win"
                stats['agent_b_wins_as_second'] += 1
            stats['agent_b_total_wins'] += 1
        else:
            # Draw
            result = "draw"
            winner_id = "draw"
            stats['draws'] += 1
        
        # End game logging
        log_file = game_logger.end_game(result, winner_id)
        logger.info(f"Game {game_num} ended: {result}, winner: {winner_id}. Log saved to {log_file}")
        
        # Train autoencoder if specified
        if args.train_autoencoder and game_num % 2 == 0:  # Train every 2 games
            logger.info("Training autoencoder on collected board states")
            memory_manager_a.train_autoencoder_on_memory(epochs=args.autoencoder_epochs)
            memory_manager_b.train_autoencoder_on_memory(epochs=args.autoencoder_epochs)
        
        # Delay between games
        if game_num < args.num_games:
            time.sleep(GAME_DELAY)
    
    # Save overall statistics
    run_id = game_logger.run_id
    stats['run_id'] = run_id
    stats_file = output_dir / f"run_{run_id}_stats_{args.board_size}x{args.board_size}_{int(time.time())}.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary statistics
    logger.info(f"\n===== Experiment Summary (Board Size: {args.board_size}x{args.board_size}) =====")
    logger.info(f"Total games: {args.num_games}")
    logger.info(f"Agent A wins as first: {stats['agent_a_wins_as_first']} ({stats['agent_a_wins_as_first']/args.num_games*100:.1f}%)")
    logger.info(f"Agent A wins as second: {stats['agent_a_wins_as_second']} ({stats['agent_a_wins_as_second']/args.num_games*100:.1f}%)")
    logger.info(f"Agent A total wins: {stats['agent_a_total_wins']} ({stats['agent_a_total_wins']/args.num_games*100:.1f}%)")
    logger.info(f"Agent B wins as first: {stats['agent_b_wins_as_first']} ({stats['agent_b_wins_as_first']/args.num_games*100:.1f}%)")
    logger.info(f"Agent B wins as second: {stats['agent_b_wins_as_second']} ({stats['agent_b_wins_as_second']/args.num_games*100:.1f}%)")
    logger.info(f"Agent B total wins: {stats['agent_b_total_wins']} ({stats['agent_b_total_wins']/args.num_games*100:.1f}%)")
    logger.info(f"Draws: {stats['draws']} ({stats['draws']/args.num_games*100:.1f}%)")
    logger.info(f"Agent A total tokens: {stats['agent_a_total_tokens']}")
    logger.info(f"Agent B total tokens: {stats['agent_b_total_tokens']}")
    logger.info(f"Agent A memory calls: {sum(stats['agent_a_memory_calls'].values())}")
    logger.info(f"Agent B memory calls: {sum(stats['agent_b_memory_calls'].values())}")
    logger.info(f"Agent Mode: {args.agent_mode}")

    # Report first player advantage
    first_player_wins = stats['agent_a_wins_as_first'] + stats['agent_b_wins_as_first']
    second_player_wins = stats['agent_a_wins_as_second'] + stats['agent_b_wins_as_second']
    logger.info(f"First player wins: {first_player_wins} ({first_player_wins/args.num_games*100:.1f}%)")
    logger.info(f"Second player wins: {second_player_wins} ({second_player_wins/args.num_games*100:.1f}%)")

    logger.info(f"Statistics saved to {stats_file}")

if __name__ == "__main__":
    main() 