import json
import os
from datetime import datetime
from pathlib import Path
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GameLogger:
    def __init__(self, logs_dir="experiments/results/game_logs"):
        """Initialize logger with directory for game logs"""
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Current game state
        self.current_game = None
        self.current_turn = None
        
        # Generate a unique run ID based on timestamp
        self.run_id = int(time.time())
        
        # Store move information temporarily
        self.last_move_info = {}
        
        logger.info(f"GameLogger initialized with logs directory: {logs_dir}")
        logger.info(f"Run ID for this session: {self.run_id}")
    
    def start_game(self, game_id, agent_a_objective, agent_b_objective, board_size=3):
        """Start a new game log"""
        self.current_game = {
            "run_id": self.run_id,
            "game_id": game_id,
            "timestamp": datetime.now().isoformat(),
            "board_size": f"{board_size}x{board_size}",
            "agent_a": {
                "objective": agent_a_objective,
                "total_tokens_used": 0,
                "memory_usage_summary": {
                    "graph_store": 0,
                    "graph_read": 0,
                    "vector_store": 0,
                    "vector_read": 0,
                    "semantic_store": 0,
                    "semantic_read": 0,
                    "update_graph_schema": 0,
                    "update_vector_schema": 0,
                    "update_semantic_schema": 0
                },
                "schema_updates": [],
                "turns": []
            },
            "agent_b": {
                "objective": agent_b_objective,
                "total_tokens_used": 0,
                "memory_usage_summary": {
                    "graph_store": 0,
                    "graph_read": 0,
                    "vector_store": 0,
                    "vector_read": 0,
                    "semantic_store": 0,
                    "semantic_read": 0,
                    "update_graph_schema": 0,
                    "update_vector_schema": 0,
                    "update_semantic_schema": 0
                },
                "schema_updates": [],
                "turns": []
            },
            "final_result": None
        }
        
        logger.info(f"Started new game log for game ID: {game_id} with board size {board_size}x{board_size}")
    
    def log_turn(self, turn_number, agent_id, board_state, action=None, 
                 memory_function=None, memory_content=None, memory_query=None,
                 schema_update=None, schema_description=None, 
                 rationale_excerpt=None, tokens_used=0):
        """Log a turn in the current game"""
        if self.current_game is None:
            logger.error("Cannot log turn: No active game")
            return False
        
        agent_key = agent_id  # 'agent_a' or 'agent_b'
        
        # Create turn data
        turn_data = {
            "turn": turn_number,
            "board_state": board_state,
            "tokens_used": tokens_used
        }
        
        # Add action if provided
        if action is not None:
            turn_data["action"] = action
        
        # Add memory function details if provided
        if memory_function is not None:
            turn_data["memory_function"] = memory_function
            
            # Update memory usage summary
            if memory_function in self.current_game[agent_key]["memory_usage_summary"]:
                self.current_game[agent_key]["memory_usage_summary"][memory_function] += 1
            
            # Add memory content or query depending on function type
            if memory_function.endswith("_store") and memory_content is not None:
                turn_data["memory_content_stored"] = memory_content
            elif memory_function.endswith("_read") and memory_query is not None:
                turn_data["memory_query"] = memory_query
        
        # Add schema update if provided
        if schema_update is not None and schema_description is not None:
            turn_data["schema_update"] = schema_update
            turn_data["schema_description"] = schema_description
            
            # Add to schema updates
            schema_update_data = {
                "turn": turn_number,
                "memory_type": schema_update.replace("update_", "").replace("_schema", "").capitalize() + "Memory",
                "new_schema_description": schema_description
            }
            
            if rationale_excerpt is not None:
                schema_update_data["rationale_excerpt"] = rationale_excerpt
            
            self.current_game[agent_key]["schema_updates"].append(schema_update_data)
            
            # Update memory usage summary for schema updates
            if schema_update in self.current_game[agent_key]["memory_usage_summary"]:
                self.current_game[agent_key]["memory_usage_summary"][schema_update] += 1
        
        # Add rationale if provided
        if rationale_excerpt is not None:
            turn_data["reasoning_excerpt"] = rationale_excerpt
        
        # Update total tokens used
        self.current_game[agent_key]["total_tokens_used"] += tokens_used
        
        # Add turn data to game log
        self.current_game[agent_key]["turns"].append(turn_data)
        
        logger.info(f"Logged turn {turn_number} for agent {agent_id}")
        return True
    
    def end_game(self, result, winner=None):
        """End the current game and save the log"""
        if self.current_game is None:
            logger.error("Cannot end game: No active game")
            return False
        
        # Set final result
        self.current_game["final_result"] = result
        if winner is not None:
            self.current_game["winner"] = winner
        
        # Save game log to file
        game_id = self.current_game["game_id"]
        board_size = self.current_game["board_size"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = self.current_game["run_id"]
        log_filename = f"run_{run_id}_game_{game_id}_{board_size}_{timestamp}.json"
        log_filepath = self.logs_dir / log_filename
        
        with open(log_filepath, 'w') as f:
            json.dump(self.current_game, f, indent=2)
        
        logger.info(f"Game {game_id} ({board_size}) ended with result: {result}. Log saved to {log_filepath}")
        
        # Clear current game
        self.current_game = None
        
        return log_filepath
    
    def extract_rationale(self, agent_response):
        """Extract rationale from agent response for logging"""
        # Simple extractor - in a real implementation, this would be more sophisticated
        # to extract the reasoning part from the agent's response
        if isinstance(agent_response, str) and len(agent_response) > 0:
            # Take the first portion as a simple heuristic
            excerpt_length = min(200, len(agent_response))
            return agent_response[:excerpt_length]
        return None

    def log_move(self, agent_id, move, move_info):
        """Log a move made by an agent, storing information for later use in log_turn"""
        if move_info is None:
            move_info = {}
        
        # Store the move info for this agent
        self.last_move_info[agent_id] = {
            "move": move,
            "move_info": move_info
        }
        
        logger.info(f"Stored move info for agent {agent_id}: {move}")
        return True 