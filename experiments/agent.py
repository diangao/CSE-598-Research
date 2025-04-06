import os
import json
import logging
import re
import time
import sys
from pathlib import Path
import openai
from typing import Dict, List, Tuple, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path("experiments/.env"))

# Get retry delay setting from environment variables
RETRY_INIT_DELAY = float(os.getenv("RETRY_INIT_DELAY", "2.0"))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verify if API key is set
if not os.getenv("OPENAI_API_KEY"):
    logger.warning("OPENAI_API_KEY not found in environment variables or .env file")
else:
    logger.info("OPENAI_API_KEY found, using it for API calls")
    # Set API key for openai
    openai.api_key = os.getenv("OPENAI_API_KEY")

class TicTacToeAgent:
    def __init__(self, agent_id, memory_manager):
        """Initialize TicTacToe agent with OpenAI API integration and memory manager"""
        self.agent_id = agent_id  # 'agent_a' or 'agent_b'
        self.memory_manager = memory_manager
        self.model = "gpt-3.5-turbo-16k"  # Changed to 16k version for higher context limit and lower cost
        
        # Load system prompt
        prompt_path = Path(f"experiments/prompts/system_{agent_id}.txt")
        if not prompt_path.exists():
            raise FileNotFoundError(f"System prompt file not found: {prompt_path}")
        
        with open(prompt_path, 'r') as f:
            self.system_prompt = f.read()
        
        # Set up function definitions for the agent
        self.functions = self._setup_functions()
        
        # Message history for the agent
        self.message_history = [{"role": "system", "content": self.system_prompt}]
        
        logger.info(f"TicTacToe agent '{agent_id}' initialized")
    
    def _setup_functions(self) -> List[Dict[str, Any]]:
        """Set up function definitions for OpenAI API"""
        return [
            {
                "name": "graph_store",
                "description": "Store structured or sequential info in GraphMemory. Great for representing transitions or causal reasoning.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "any structured info you consider useful later"
                        }
                    },
                    "required": ["content"]
                }
            },
            {
                "name": "graph_read",
                "description": "Retrieve from GraphMemory based on relational or sequential similarity.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "describe your current reasoning context"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "vector_store",
                "description": "Store pattern-like or case-based info in VectorMemory. Best for similarity search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "any experience you'd like to reuse via fuzzy matching"
                        }
                    },
                    "required": ["content"]
                }
            },
            {
                "name": "vector_read",
                "description": "Retrieve from VectorMemory based on vector similarity.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "describe the current situation"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "semantic_store",
                "description": "Store ideas or concepts in SemanticMemory. Focused on language-level meaning.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "describe any strategic idea, logic, or concept"
                        }
                    },
                    "required": ["content"]
                }
            },
            {
                "name": "semantic_read",
                "description": "Retrieve from SemanticMemory based on conceptual or linguistic similarity.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "explain your current challenge or intent"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "update_graph_schema",
                "description": "Update how GraphMemory organizes and links experiences. You may redefine what a node or edge means, or how states relate to actions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "new_schema_description": {
                            "type": "string",
                            "description": "Describe the new format of how you want GraphMemory to represent experiences"
                        }
                    },
                    "required": ["new_schema_description"]
                }
            },
            {
                "name": "update_vector_schema",
                "description": "Change how content is embedded or labeled in VectorMemory. You may modify how you group, tag, or search memories.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "new_schema_description": {
                            "type": "string",
                            "description": "Describe your revised strategy for how you want to encode or retrieve vector memory content"
                        }
                    },
                    "required": ["new_schema_description"]
                }
            },
            {
                "name": "update_semantic_schema",
                "description": "Modify how you conceptually structure SemanticMemory. You may shift the abstraction level, include reasoning strategies, or organize by intent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "new_schema_description": {
                            "type": "string",
                            "description": "Explain your updated plan for organizing conceptual or strategic ideas in memory"
                        }
                    },
                    "required": ["new_schema_description"]
                }
            }
        ]
    
    def reset(self):
        """Reset agent state for a new game"""
        # Reset message history to only include system prompt
        self.message_history = [{"role": "system", "content": self.system_prompt}]
        logger.info(f"Reset agent {self.agent_id} for a new game")
    
    def get_move(self, board_state) -> Tuple[Optional[Tuple[int, int]], Dict[str, Any]]:
        """Get the next move from the agent, allowing for memory function calls first"""
        # Add board state to message history
        self.message_history.append({
            "role": "user", 
            "content": f"Current board state:\n{board_state}\n\nWhat's your next move?"
        })
        
        logger.info(f"Requesting move from agent {self.agent_id}")
        
        # Variables to track the agent's response
        move = None
        memory_function_used = None
        memory_content = None
        memory_query = None
        schema_update = None
        schema_description = None
        tokens_used = 0
        raw_response = None
        
        # Call the OpenAI API to get the agent's response
        try:
            # Add exponential backoff retry logic
            max_retries = 5
            retry_delay = RETRY_INIT_DELAY
            
            for retry_attempt in range(max_retries):
                try:
                    response = openai.chat.completions.create(
                        model=self.model,
                        messages=self.message_history,
                        functions=self.functions,
                        function_call="auto",
                        temperature=0.7
                    )
                    # If we get here, the API call was successful
                    break
                except openai.RateLimitError as e:
                    if retry_attempt < max_retries - 1:
                        sleep_time = retry_delay * (2 ** retry_attempt)  # Exponential backoff
                        logger.warning(f"Rate limit exceeded, retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
                    else:
                        raise  # Re-raise if we've exhausted our retries
                except openai.InsufficientQuotaError as e:
                    logger.error(f"API quota insufficient error: Your OpenAI API quota has been depleted. Please check your billing and plan details.")
                    logger.error(f"Detailed error message: {str(e)}")
                    # Return a random move to let the game continue
                    move = None
                    return move, {
                        "tokens_used": 0,
                        "memory_function": None,
                        "memory_content": None,
                        "memory_query": None,
                        "schema_update": None,
                        "schema_description": None,
                        "raw_response": None
                    }
            
            # Track token usage
            tokens_used = response.usage.total_tokens
            raw_response = response.choices[0].message
            
            # Handle response based on whether it used a function or made a direct move
            assistant_message = response.choices[0].message
            self.message_history.append(assistant_message)
            
            # Check if the assistant used a function
            if hasattr(assistant_message, 'function_call') and assistant_message.function_call:
                function_name = assistant_message.function_call.name
                function_args = json.loads(assistant_message.function_call.arguments)
                
                # Handle different memory functions
                if function_name == "graph_store":
                    content = function_args.get("content")
                    result = self.memory_manager.graph_store(content)
                    memory_function_used = "graph_store"
                    memory_content = content
                
                elif function_name == "graph_read":
                    query = function_args.get("query")
                    result = self.memory_manager.graph_read(query)
                    memory_function_used = "graph_read"
                    memory_query = query
                
                elif function_name == "vector_store":
                    content = function_args.get("content")
                    result = self.memory_manager.vector_store(content)
                    memory_function_used = "vector_store"
                    memory_content = content
                
                elif function_name == "vector_read":
                    query = function_args.get("query")
                    result = self.memory_manager.vector_read(query)
                    memory_function_used = "vector_read"
                    memory_query = query
                
                elif function_name == "semantic_store":
                    content = function_args.get("content")
                    result = self.memory_manager.semantic_store(content)
                    memory_function_used = "semantic_store"
                    memory_content = content
                
                elif function_name == "semantic_read":
                    query = function_args.get("query")
                    result = self.memory_manager.semantic_read(query)
                    memory_function_used = "semantic_read"
                    memory_query = query
                
                elif function_name == "update_graph_schema":
                    new_schema = function_args.get("new_schema_description")
                    result = self.memory_manager.update_graph_schema(new_schema)
                    memory_function_used = "update_graph_schema"
                    schema_update = "update_graph_schema"
                    schema_description = new_schema
                
                elif function_name == "update_vector_schema":
                    new_schema = function_args.get("new_schema_description")
                    result = self.memory_manager.update_vector_schema(new_schema)
                    memory_function_used = "update_vector_schema"
                    schema_update = "update_vector_schema"
                    schema_description = new_schema
                
                elif function_name == "update_semantic_schema":
                    new_schema = function_args.get("new_schema_description")
                    result = self.memory_manager.update_semantic_schema(new_schema)
                    memory_function_used = "update_semantic_schema"
                    schema_update = "update_semantic_schema"
                    schema_description = new_schema
                
                # Add function result to message history
                self.message_history.append({
                    "role": "function",
                    "name": function_name,
                    "content": str(result)
                })
                
                # Now get the actual move after using memory function
                return self.get_move(board_state)
            
            else:
                # Extract move from response content
                content = assistant_message.content
                # Try to parse JSON move format: {"move": [row, col]}
                try:
                    move_match = re.search(r'{"move":\s*\[(\d+),\s*(\d+)\]}', content)
                    if move_match:
                        row, col = int(move_match.group(1)), int(move_match.group(2))
                        move = (row, col)
                    else:
                        # Fallback to more flexible parsing if JSON format is not found
                        move_pattern = r'\[(\d+)[, ]+(\d+)\]'
                        move_match = re.search(move_pattern, content)
                        if move_match:
                            row, col = int(move_match.group(1)), int(move_match.group(2))
                            move = (row, col)
                        else:
                            logger.warning(f"Could not parse move from agent {self.agent_id}'s response: {content}")
                
                except Exception as e:
                    logger.error(f"Error parsing move: {e}")
                    logger.error(f"Response was: {content}")
                    move = None
        
        except Exception as e:
            logger.error(f"Error getting move from agent {self.agent_id}: {e}")
            move = None
        
        # Return the move and additional information for logging
        return move, {
            "tokens_used": tokens_used,
            "memory_function": memory_function_used,
            "memory_content": memory_content,
            "memory_query": memory_query,
            "schema_update": schema_update,
            "schema_description": schema_description,
            "raw_response": raw_response
        } 