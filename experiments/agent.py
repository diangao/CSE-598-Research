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
        """Initialize agent with its ID and memory manager"""
        self.agent_id = agent_id
        self.memory_manager = memory_manager
        
        # Load the appropriate system prompt
        if agent_id == "agent_a":
            system_prompt_file = "experiments/prompts/system_agent_a.txt"
        else:
            system_prompt_file = "experiments/prompts/system_agent_b.txt"
        
        with open(system_prompt_file, "r") as f:
            system_prompt = f.read()
            
        # Save original system prompt
        self.system_prompt = system_prompt
            
        # Add memory constraint information to the system prompt if applicable
        memory_constraint = memory_manager.memory_constraint
        if memory_constraint != "none":
            # Prepare constraint message based on the type
            if memory_constraint == "graph_only":
                constraint_msg = "\n\nIMPORTANT: For this game, you are ONLY allowed to use GraphMemory functions (graph_store, graph_read, update_graph_schema).\nVectorMemory and SemanticMemory are disabled and any attempt to use them will fail."
            elif memory_constraint == "vector_only":
                constraint_msg = "\n\nIMPORTANT: For this game, you are ONLY allowed to use VectorMemory functions (vector_store, vector_read, update_vector_schema).\nGraphMemory and SemanticMemory are disabled and any attempt to use them will fail."
            else:
                constraint_msg = f"\n\nNOTE: Memory constraint '{memory_constraint}' is in effect for this game."
            
            # Add the constraint message to the system prompt
            system_prompt += constraint_msg
            logging.info(f"Added memory constraint info to {agent_id} prompt: {memory_constraint}")
            
        # Set up OpenAI API
        self.model = os.getenv('OPENAI_MODEL', "gpt-4o")
        
        # Define memory-related functions the agent can use
        self.functions = [
            {
                "name": "graph_store",
                "description": "Store information in graph memory for future reference",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Content to store in graph memory (e.g., board state, move, outcome)"
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
            },
            {
                "name": "make_move",
                "description": "Make a move on the TicTacToe board.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "move": {
                            "type": "array",
                            "items": {
                                "type": "integer",
                                "description": "Row and column index (e.g., [row, col])"
                            },
                            "description": "The move to make, represented as [row, col]."
                        }
                    },
                    "required": ["move"]
                }
            }
        ]
        
        # Message history for the agent
        self.message_history = [{"role": "system", "content": system_prompt}]
        
        logger.info(f"TicTacToe agent '{agent_id}' initialized")
    
    def reset(self):
        """Reset agent state for a new game"""
        # Reset message history to only include system prompt
        self.message_history = [{"role": "system", "content": self.system_prompt}]
        logger.info(f"Reset agent {self.agent_id} for a new game")
    
    def get_move(self, board_state, depth=0) -> Tuple[Optional[Tuple[int, int]], Dict[str, Any]]:
        """Get the next move from the agent, allowing for memory function calls first"""
        # Add recursion depth limit to prevent infinite loops
        MAX_DEPTH = 3
        if depth >= MAX_DEPTH:
            logger.warning(f"Maximum memory function call depth reached ({MAX_DEPTH}), forcing move decision")
            
            # Get the content of the last user-added message to determine the last memory function used
            last_function_name = None
            for msg in reversed(self.message_history):
                if msg.get("role") == "function":
                    last_function_name = msg.get("name")
                    break
            
            # Force return a move instead of continuing recursion
            return None, {
                "tokens_used": 0,
                "memory_function": last_function_name, # Pass the name of the last memory function used
                "memory_content": None,
                "memory_query": None,
                "schema_update": None,
                "schema_description": None,
                "raw_response": "Maximum recursion depth reached, need to make a move"
            }
        
        # Add stronger warning if we're getting close to the depth limit
        memory_warning = ""
        if depth == MAX_DEPTH - 1:
            memory_warning = "IMPORTANT: You MUST make a move now. Do not use any more memory functions. Return your move directly as {\"move\": [row, col]}."
        
        # Add board state to message history
        self.message_history.append({
            "role": "user", 
            "content": f"Current board state:\n{board_state}\n\nWhat's your next move? {memory_warning}"
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
                    
                    # Get the content of the last user-added message to determine the last memory function used
                    last_function_name = None
                    for msg in reversed(self.message_history):
                        if msg.get("role") == "function":
                            last_function_name = msg.get("name")
                            break
                            
                    # Return a random move to let the game continue
                    move = None
                    raw_response = "Quota insufficient error - API call failed"
                    memory_function_used = last_function_name
                    # Continue with the code below to fall back to a random move
                except Exception as e:
                    logger.error(f"Error getting move from agent {self.agent_id}: {e}")
                    move = None
                    raw_response = f"API error: {str(e)}"
                    # Continue with the code below to fall back to a random move
            
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
                
                # Add an explicit instruction to provide a move after using memory
                next_instruction = f"You've used the memory function: {function_name}. Now please make a specific move decision based on the current board state."
                if depth >= MAX_DEPTH - 2:  # Add stronger instruction when nearing depth limit
                    next_instruction += " IMPORTANT: You MUST make a move now without using any more memory functions. Return your move as {\"move\": [row, col]}."
                
                self.message_history.append({
                    "role": "user",
                    "content": next_instruction
                })
                
                # Now get the actual move after using memory function, incrementing depth
                logger.info(f"Agent {self.agent_id} used memory function {function_name}, now requesting actual move (depth {depth+1})")
                next_move, next_info = self.get_move(board_state, depth + 1)
                
                if next_move is not None:
                    next_info["memory_function"] = function_name
                    if function_name.endswith("_store"):
                        next_info["memory_content"] = function_args.get("content", "")
                    elif function_name.endswith("_read"):
                        next_info["memory_query"] = function_args.get("query", "")
                    elif "schema" in function_name:
                        next_info["schema_update"] = function_name
                        next_info["schema_description"] = function_args.get("new_schema_description", "")
                    
                    return next_move, next_info
                
                return next_move, next_info
            
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

    def submit_board(self, state: str, validator=None):
        """Alternative method to get move using simpler interaction pattern.
           This implementation is inspired by project_test approach but adapted to our framework.
        """
        MAX_TRIES = 3
        # Add board state to message history
        self.message_history.append({
            "role": "user", 
            "content": f"Current board state:\n{state}\n\nRetrieve from memory if needed or make a move."
        })
        
        logger.info(f"Requesting move from agent {self.agent_id} using simplified pattern")
        

        memory_function_used = None
        memory_content = None
        memory_query = None
        schema_update = None
        schema_description = None
        tokens_used = 0
        raw_response = None
        
        # Function to process response and extract move
        def process_response(response):
            nonlocal memory_function_used, memory_content, memory_query, tokens_used, raw_response
            
            # Track token usage
            tokens_used += response.usage.total_tokens
            
            # Store raw response for logging
            assistant_message = response.choices[0].message
            raw_response = assistant_message.content
            self.message_history.append(assistant_message)
            
            # Check if the assistant used a function
            if hasattr(assistant_message, 'function_call') and assistant_message.function_call:
                function_name = assistant_message.function_call.name
                function_args = json.loads(assistant_message.function_call.arguments)
                
                # If the function is make_move, extract the move directly
                if function_name == "make_move":
                    move_coords = function_args.get("move")
                    if move_coords and len(move_coords) == 2:
                        return True, (move_coords[0], move_coords[1])
                
                # If it's a memory function, process it
                result = None
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
                
                # Add function result to message history
                if result:
                    self.message_history.append({
                        "role": "function",
                        "name": function_name,
                        "content": str(result)
                    })
                    # We need another round since this was a memory function, not a move
                    return False, result
            
            # Try to parse a move from the text response if we didn't get a function call
            content = assistant_message.content
            move_match = re.search(r'{"move":\s*\[(\d+),\s*(\d+)\]}', content)
            if move_match:
                row, col = int(move_match.group(1)), int(move_match.group(2))
                return True, (row, col)
            
            # Fallback to more flexible parsing if JSON format is not found
            move_pattern = r'\[(\d+)[, ]+(\d+)\]'
            move_match = re.search(move_pattern, content)
            if move_match:
                row, col = int(move_match.group(1)), int(move_match.group(2))
                return True, (row, col)
            
            # If we couldn't extract a move, return the content so we can prompt again
            logger.warning(f"Could not extract move from agent response: {content[:100]}...")
            return False, content
        
        # First attempt to get a move
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=self.message_history,
                functions=self.functions,
                function_call="auto",
                temperature=0.7
            )
            success, data = process_response(response)
        except Exception as e:
            logger.error(f"Error in API call: {e}")
            return (None, {
                "tokens_used": tokens_used,
                "memory_function": memory_function_used,
                "memory_content": memory_content,
                "memory_query": memory_query,
                "schema_update": schema_update,
                "schema_description": schema_description,
                "raw_response": raw_response,
                "error": str(e)
            })
        
        # If we got a memory function response, prompt for a move now
        num_tries = 0
        while not success and num_tries < MAX_TRIES:
            if isinstance(data, str):
                self.message_history.append({
                    "role": "user",
                    "content": "You need to make a move now. Please return your move as {\"move\": [row, col]}"
                })
            else:
                self.message_history.append({
                    "role": "user",
                    "content": "You've used memory. Now please make a move. Return your move as {\"move\": [row, col]}"
                })
            
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=self.message_history,
                    functions=self.functions,
                    function_call="auto",
                    temperature=0.7
                )
                success, data = process_response(response)
            except Exception as e:
                logger.error(f"Error in API call: {e}")
                return (None, {
                    "tokens_used": tokens_used,
                    "memory_function": memory_function_used,
                    "memory_content": memory_content,
                    "memory_query": memory_query,
                    "schema_update": schema_update,
                    "schema_description": schema_description,
                    "raw_response": raw_response,
                    "error": str(e)
                })
            
            num_tries += 1
        
        # After MAX_TRIES, force a final attempt with a stronger message
        if not success:
            self.message_history.append({
                "role": "user",
                "content": "IMPORTANT: You MUST make a move now. Do not use any more memory functions. Make a specific move by returning {\"move\": [row, col]}"
            })
            
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=self.message_history,
                    functions=[f for f in self.functions if f["name"] == "make_move"],  # Only allow make_move
                    function_call={"name": "make_move"},  # Force make_move
                    temperature=0.7
                )
                success, data = process_response(response)
            except Exception as e:
                logger.error(f"Error in final API call: {e}")
                return (None, {
                    "tokens_used": tokens_used,
                    "memory_function": memory_function_used,
                    "memory_content": memory_content,
                    "memory_query": memory_query,
                    "schema_update": schema_update,
                    "schema_description": schema_description,
                    "raw_response": raw_response,
                    "error": str(e)
                })
        
        # Validate the move if a validator is provided
        if validator and success and not validator(data[0], data[1]):
            num_tries = 0
            while not validator(data[0], data[1]) and num_tries < MAX_TRIES:
                self.message_history.append({
                    "role": "user",
                    "content": f"Invalid move: {data}. Please provide a valid move."
                })
                
                try:
                    response = openai.chat.completions.create(
                        model=self.model,
                        messages=self.message_history,
                        functions=[f for f in self.functions if f["name"] == "make_move"],
                        function_call={"name": "make_move"},
                        temperature=0.7
                    )
                    success, data = process_response(response)
                except Exception as e:
                    logger.error(f"Error in validation API call: {e}")
                    # Fall back to a random valid move instead of crashing
                    return (None, {
                        "tokens_used": tokens_used,
                        "memory_function": memory_function_used,
                        "memory_content": memory_content,
                        "memory_query": memory_query,
                        "schema_update": schema_update,
                        "schema_description": schema_description,
                        "raw_response": raw_response,
                        "error": str(e)
                    })
                
                num_tries += 1
                # 如果尝试多次后仍无法获取有效移动，success会被设置为False
        
        # 返回与get_move相同格式的结果
        if success:
            return (data, {
                "tokens_used": tokens_used,
                "memory_function": memory_function_used,
                "memory_content": memory_content,
                "memory_query": memory_query,
                "schema_update": schema_update,
                "schema_description": schema_description,
                "raw_response": raw_response
            })
        else:
            return (None, {
                "tokens_used": tokens_used,
                "memory_function": memory_function_used,
                "memory_content": memory_content,
                "memory_query": memory_query,
                "schema_update": schema_update,
                "schema_description": schema_description,
                "raw_response": raw_response
            }) 