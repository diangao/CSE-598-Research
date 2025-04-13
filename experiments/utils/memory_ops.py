import json
import os
from pathlib import Path
import logging
import torch
import numpy as np

# Import the autoencoder and memory classes
from experiments.utils.autoencoder import AutoEncoder, board_to_tensor, encode_board, find_similar_boards
from experiments.utils.graph_memory import GraphMemory
from experiments.utils.vector_memory import VectorizedMemory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, agent_id, memory_base_dir="experiments/agents", memory_constraint="none", use_pretrained_autoencoder=False, pretrained_autoencoder_path=None):
        """Initialize memory manager for an agent"""
        self.agent_id = agent_id
        self.memory_dir = Path(f"{memory_base_dir}/{agent_id}/memory")
        
        # Set memory constraints
        self.memory_constraint = memory_constraint
        logger.info(f"Memory constraint for agent {agent_id}: {memory_constraint}")
        
        # Create memory files if they don't exist
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize memory files with empty data
        self.graph_memory_path = self.memory_dir / "graph_memory.json"
        self.vector_memory_path = self.memory_dir / "vector_memory.json"
        self.semantic_memory_path = self.memory_dir / "semantic_memory.json"
        self.schema_path = self.memory_dir / "schemas.json"
        self.ae_model_path = self.memory_dir / "autoencoder.pt"
        self.graph_pickle_path = self.memory_dir / "graph_memory.pkl"
        
        # Initialize autoencoder for vector embeddings
        self.autoencoder = AutoEncoder()
        
        # Handle pretrained autoencoder if specified
        self.use_pretrained_autoencoder = use_pretrained_autoencoder
        self.pretrained_autoencoder_path = Path(pretrained_autoencoder_path) if pretrained_autoencoder_path else None
        
        if use_pretrained_autoencoder and pretrained_autoencoder_path and Path(pretrained_autoencoder_path).exists():
            try:
                self.autoencoder.load_state_dict(torch.load(pretrained_autoencoder_path))
                logger.info(f"Loaded pretrained autoencoder model from {pretrained_autoencoder_path}")
            except Exception as e:
                logger.error(f"Error loading pretrained autoencoder: {e}")
                logger.info("Falling back to standard autoencoder initialization")
                self.load_or_init_autoencoder()
        else:
            # Standard initialization
            self.load_or_init_autoencoder()
        
        # Initialize GraphMemory for direct graph operations
        self.graph_memory = GraphMemory(storage=str(self.graph_pickle_path))
        
        # Initialize VectorizedMemory using the autoencoder
        self.vector_memory = VectorizedMemory(self.autoencoder)
        
        # Initialize memory files if they don't exist
        self._initialize_memory_files()
        
        logger.info(f"Memory manager initialized for agent {agent_id}")
    
    def load_or_init_autoencoder(self):
        """Load existing autoencoder or initialize a new one"""
        if self.ae_model_path.exists():
            try:
                self.autoencoder.load_state_dict(torch.load(self.ae_model_path))
                logger.info(f"Loaded autoencoder model for agent {self.agent_id}")
            except Exception as e:
                logger.error(f"Error loading autoencoder: {e}")
                logger.info("Initializing new autoencoder")
                self.autoencoder = AutoEncoder()
        else:
            logger.info(f"No existing autoencoder found for agent {self.agent_id}, using default")
    
    def save_autoencoder(self):
        """Save the current autoencoder state"""
        try:
            torch.save(self.autoencoder.state_dict(), self.ae_model_path)
            logger.info(f"Saved autoencoder model for agent {self.agent_id}")
        except Exception as e:
            logger.error(f"Error saving autoencoder: {e}")
    
    def train_autoencoder_on_memory(self, epochs=500):
        """Train the autoencoder on all stored board states"""
        # Collect board states from graph memory
        board_states = []
        
        # Get board states from GraphMemory
        for node_id, state in self.graph_memory.nodes.items():
            if isinstance(state, str) and len(state) == 9 and all(c in "XO-" for c in state):
                board_states.append(state)
                logger.info(f"Found board state in GraphMemory: {state}")
        
        # Also get from JSON format
        with open(self.graph_memory_path, 'r') as f:
            graph_memory_json = json.load(f)
        
        for entry in graph_memory_json.get("entries", []):
            content = entry.get("content", "")
            # First try to extract board state from content directly
            clean_content = content.replace(" ", "").replace("\n", "")
            
            # Try to find a 9-character sequence of X, O, and -
            import re
            board_matches = re.findall(r'[XO\-]{9}', clean_content)
            if board_matches:
                for match in board_matches:
                    if match not in board_states:
                        board_states.append(match)
                        logger.info(f"Found board state via regex: {match}")
                continue
            
            # Try the line-by-line approach if regex doesn't work
            for line in content.split("\n"):
                cleaned = line.replace(" ", "").replace("\n", "")
                if len(cleaned) == 9 and all(c in "XO-" for c in cleaned):
                    if cleaned not in board_states:
                        board_states.append(cleaned)
                        logger.info(f"Found board state via line parsing: {cleaned}")
        
        # Check vector memory as well
        with open(self.vector_memory_path, 'r') as f:
            vector_memory = json.load(f)
        
        for entry in vector_memory.get("entries", []):
            if "board_state" in entry and entry["board_state"] not in board_states:
                board_states.append(entry["board_state"])
                logger.info(f"Found board state in vector memory: {entry['board_state']}")
            
            # Also try to extract from content
            content = entry.get("content", "")
            clean_content = content.replace(" ", "").replace("\n", "")
            
            # Look for 9-character sequence of X, O, and -
            board_matches = re.findall(r'[XO\-]{9}', clean_content)
            if board_matches:
                for match in board_matches:
                    if match not in board_states:
                        board_states.append(match)
                        logger.info(f"Found board state via regex in vector memory: {match}")
        
        # Check if we have more than 9 states in total (to make sure we have enough for training)
        # If not, generate some synthetic ones
        if len(board_states) < 9:
            logger.info(f"Only found {len(board_states)} board states. Adding synthetic ones for training.")
            # Add empty board
            if "---------" not in board_states:
                board_states.append("---------")
            
            # Add some common patterns if not already there
            common_patterns = [
                "X--------", "----X----", "--------X",  # Corners and center
                "X---O----", "----XO---", "-----X--O",  # Some simple patterns
                "XOX------", "X--O--X--", "XO--X----"   # More complex patterns
            ]
            
            for pattern in common_patterns:
                if pattern not in board_states:
                    board_states.append(pattern)
        
        if not board_states:
            logger.info("No board states found for training the autoencoder")
            return False
        
        logger.info(f"Training autoencoder on {len(board_states)} board states")
        for state in board_states[:5]:  # Log first 5 states for debugging
            logger.info(f"Example state: {state}")
        
        # Import the training function
        from experiments.utils.autoencoder import train_encoder
        
        # Train the model
        train_encoder(self.autoencoder, board_states, epochs=epochs)
        
        # Save the updated model
        self.save_autoencoder()
        
        # Reinitialize VectorizedMemory with updated autoencoder
        self.vector_memory = VectorizedMemory(self.autoencoder)
        
        return True
    
    def _initialize_memory_files(self):
        """Initialize memory files with empty data if they don't exist"""
        # Initialize graph memory JSON (for backward compatibility)
        if not self.graph_memory_path.exists():
            with open(self.graph_memory_path, 'w') as f:
                json.dump({"entries": []}, f)
        
        # Initialize vector memory
        if not self.vector_memory_path.exists():
            with open(self.vector_memory_path, 'w') as f:
                json.dump({"entries": []}, f)
        
        # Initialize semantic memory
        if not self.semantic_memory_path.exists():
            with open(self.semantic_memory_path, 'w') as f:
                json.dump({"entries": []}, f)
        
        # Initialize schemas
        if not self.schema_path.exists():
            with open(self.schema_path, 'w') as f:
                json.dump({
                    "graph_schema": "Default graph schema: nodes represent board states, edges represent moves",
                    "vector_schema": "Default vector schema: memories stored as board state + action pairs",
                    "semantic_schema": "Default semantic schema: conceptual strategies organized by game phase"
                }, f)
    
    def graph_store(self, content):
        """Store content in graph memory (both JSON and GraphMemory object)"""
        # Check if constrained to vector-only memory
        if self.memory_constraint == "vector_only":
            logger.info(f"Agent {self.agent_id} attempted to use graph_store but is constrained to vector_only memory")
            return "Graph memory is disabled due to vector_only constraint."
                
        # Extract board state from content if possible
        board_state = None
        next_state = None
        action = None
        
        # Simple parsing to find board state, action, and next state
        lines = content.split("\n")
        for i, line in enumerate(lines):
            # Find board state
            cleaned = line.replace(" ", "").replace("\n", "")
            if len(cleaned) == 9 and all(c in "XO-" for c in cleaned):
                if board_state is None:
                    board_state = cleaned
                elif next_state is None:
                    next_state = cleaned
            
            # Try to find action in text
            if "move" in line.lower() or "position" in line.lower():
                import re
                # Look for patterns like [0, 1] or (1, 2)
                coord_match = re.search(r'[\[\(](\d+)[,\s]+(\d+)[\]\)]', line)
                if coord_match:
                    row, col = int(coord_match.group(1)), int(coord_match.group(2))
                    action = (row, col)
        
        # Store in the GraphMemory object if we have enough info
        if board_state:
            # If we don't have all info, just store the board state with empty values
            if action is None:
                action = (-1, -1)  # Placeholder
            if next_state is None:
                next_state = board_state  # Just store same state
                
            self.graph_memory.store(board_state, action, next_state, metadata={"description": content})
            self.graph_memory.export_graph()  # Persist to disk
            logger.info(f"Stored transition in GraphMemory with state: {board_state}")
        
        # Also store in the JSON format for backward compatibility
        with open(self.graph_memory_path, 'r') as f:
            memory = json.load(f)
        
        memory["entries"].append({
            "id": len(memory["entries"]),
            "content": content,
            "timestamp": self._get_timestamp()
        })
        
        with open(self.graph_memory_path, 'w') as f:
            json.dump(memory, f, indent=2)
        
        logger.info(f"Stored in graph memory: {content[:30]}...")
        return True
    
    def graph_read(self, query):
        """Read from graph memory based on query (using both JSON and GraphMemory)"""
        # Check if constrained to vector-only memory
        if self.memory_constraint == "vector_only":
            logger.info(f"Agent {self.agent_id} attempted to use graph_read but is constrained to vector_only memory")
            return "Graph memory is disabled due to vector_only constraint."
            
        # Try to extract board state from query
        board_state = None
        for line in query.split("\n"):
            cleaned = line.replace(" ", "").replace("\n", "")
            if len(cleaned) == 9 and all(c in "XO-" for c in cleaned):
                board_state = cleaned
                break
        
        # Results from GraphMemory
        graph_results = []
        if board_state:
            transitions = self.graph_memory.retrieve(board_state)
            if transitions:
                logger.info(f"Found {len(transitions)} transitions from state {board_state} in GraphMemory")
                for edge in transitions:
                    action = edge["action"]
                    next_encoded = edge["next_state"]
                    next_state = self.graph_memory.nodes.get(next_encoded, "Unknown")
                    metadata = edge["metadata"]
                    
                    if isinstance(metadata, dict) and "description" in metadata:
                        description = metadata["description"]
                    else:
                        description = f"Move {action} from {board_state} to {next_state}"
                    
                    graph_results.append(description)
        
        # Also get from JSON format for backward compatibility
        with open(self.graph_memory_path, 'r') as f:
            memory = json.load(f)
        
        # Get the schema for context
        with open(self.schema_path, 'r') as f:
            schemas = json.load(f)
            graph_schema = schemas.get("graph_schema", "Default graph schema")
        
        entries = memory["entries"]
        
        if not entries and not graph_results:
            return f"Graph memory is empty. Current schema: {graph_schema}"
        
        result = f"Graph Memory (using schema: {graph_schema}):\n"
        
        # Add GraphMemory results first if available
        if graph_results:
            result += "Relevant transitions:\n"
            for desc in graph_results:
                result += f"- {desc}\n"
            result += "\n"
        
        # Add some entries from JSON storage (last 5 or fewer)
        result += "Recent memories:\n"
        for e in entries[-5:]:
            result += f"Entry {e['id']}: {e['content']}\n\n"
        
        logger.info(f"Read from graph memory with query: {query[:30]}...")
        return result
    
    def vector_store(self, content, board_state=None):
        """Store content in vector memory with optional explicit board state"""
        # Check if constrained to graph-only memory
        if self.memory_constraint == "graph_only":
            logger.info(f"Agent {self.agent_id} attempted to use vector_store but is constrained to graph_only memory")
            return "Vector memory is disabled due to graph_only constraint."
            
        # Try to extract board state from content
        if board_state is None:
            for line in content.split("\n"):
                if len(line.replace(" ", "").replace("\n", "")) == 9 and all(c in "XO-" for c in line.replace(" ", "").replace("\n", "")):
                    board_state = line.replace(" ", "").replace("\n", "")
                    break
        
        # Store in VectorizedMemory if we found a board state
        if board_state:
            try:
                embedding = encode_board(self.autoencoder, board_state).tolist()
                self.vector_memory.store(board_state, content, embedding)
                logger.info(f"Stored in VectorizedMemory with board state: {board_state}")
            except Exception as e:
                logger.error(f"Error storing in VectorizedMemory: {e}")
        
        # Also store in the JSON format for backward compatibility
        with open(self.vector_memory_path, 'r') as f:
            memory = json.load(f)
        
        memory["entries"].append({
            "id": len(memory["entries"]),
            "content": content,
            "board_state": board_state,
            "embedding": embedding if board_state else None,
            "timestamp": self._get_timestamp()
        })
        
        with open(self.vector_memory_path, 'w') as f:
            json.dump(memory, f, indent=2)
        
        logger.info(f"Stored in vector memory: {content[:30]}...")
        return True
    
    def vector_read(self, query):
        """Read from vector memory based on query, using autoencoder for similarity search"""
        # Check if constrained to graph-only memory
        if self.memory_constraint == "graph_only":
            logger.info(f"Agent {self.agent_id} attempted to use vector_read but is constrained to graph_only memory")
            return "Vector memory is disabled due to graph_only constraint."
            
        # Try to extract board state from query for similarity search
        board_state = None
        for line in query.split("\n"):
            cleaned = line.replace(" ", "").replace("\n", "")
            if len(cleaned) == 9 and all(c in "XO-" for c in cleaned):
                board_state = cleaned
                break
        
        # Get the schema for context
        with open(self.schema_path, 'r') as f:
            schemas = json.load(f)
            vector_schema = schemas.get("vector_schema", "Default vector schema")
        
        result = f"Vector Memory (using schema: {vector_schema}):\n"
        
        # Use VectorizedMemory if we found a board state
        if board_state:
            try:
                vector_results = self.vector_memory.retrieve(board_state, k=3)
                if vector_results:
                    result += "Most similar board states:\n"
                    for metadata in vector_results:
                        if metadata and isinstance(metadata, dict) and "content" in metadata:
                            result += f"{metadata['content'][:200]}...\n\n"
                    logger.info(f"Found {len(vector_results)} similar states using VectorizedMemory")
            except Exception as e:
                logger.error(f"Error retrieving from VectorizedMemory: {e}")
        
        # Also search JSON storage for backward compatibility
        with open(self.vector_memory_path, 'r') as f:
            memory = json.load(f)
            
        entries = memory["entries"]
        
        if not entries:
            if "Most similar board states" not in result:  # Only return empty message if no results at all
                return f"Vector memory is empty. Current schema: {vector_schema}"
            return result
        
        # If we found a board state and have entries with embeddings
        similar_entries = []
        if board_state:
            # Get entries with embeddings
            entries_with_embeddings = [e for e in entries if "embedding" in e and "board_state" in e]
            
            if entries_with_embeddings:
                try:
                    # Get query embedding
                    query_embedding = encode_board(self.autoencoder, board_state).tolist()
                    
                    # Calculate similarities
                    for entry in entries_with_embeddings:
                        entry_embedding = entry["embedding"]
                        
                        # Calculate cosine similarity
                        dot_product = sum(a*b for a, b in zip(query_embedding, entry_embedding))
                        norm_q = sum(a*a for a in query_embedding) ** 0.5
                        norm_e = sum(a*a for a in entry_embedding) ** 0.5
                        
                        if norm_q > 0 and norm_e > 0:  # Avoid division by zero
                            similarity = dot_product / (norm_q * norm_e)
                            similar_entries.append((entry, similarity))
                    
                    # Sort by similarity (highest first)
                    similar_entries.sort(key=lambda x: x[1], reverse=True)
                    
                    # Take top 3
                    similar_entries = similar_entries[:3]
                    
                    logger.info(f"Found {len(similar_entries)} similar board states in JSON storage")
                except Exception as e:
                    logger.error(f"Error in similarity search: {e}")
        
        # Add similar entries from JSON storage if we found any
        if similar_entries:
            # Only add this header if we haven't already added similar results from VectorizedMemory
            if "Most similar board states" not in result:
                result += "Most similar board states:\n"
            
            for entry, similarity in similar_entries:
                result += f"Board: {entry['board_state']} (Similarity: {similarity:.2f})\n"
                result += f"Entry {entry['id']}: {entry['content'][:100]}...\n\n"
        
        # Always include some general entries
        result += "Other recent memories:\n"
        for e in entries[-3:]:  # Last 3 entries
            result += f"Entry {e['id']}: {e['content'][:100]}...\n"
        
        logger.info(f"Read from vector memory with query: {query[:30]}...")
        return result
    
    def semantic_store(self, content):
        """Store content in semantic memory"""
        # Both graph_only and vector_only constraints disallow semantic memory
        if self.memory_constraint in ["graph_only", "vector_only"]:
            logger.info(f"Agent {self.agent_id} attempted to use semantic_store but is constrained to {self.memory_constraint}")
            return "Semantic memory is disabled due to memory constraints."
            
        with open(self.semantic_memory_path, 'r') as f:
            memory = json.load(f)
        
        # Add to entries
        memory["entries"].append({
            "id": len(memory["entries"]),
            "content": content,
            "timestamp": self._get_timestamp()
        })
        
        with open(self.semantic_memory_path, 'w') as f:
            json.dump(memory, f, indent=2)
        
        logger.info(f"Stored in semantic memory: {content[:30]}...")
        return True
    
    def semantic_read(self, query):
        """Read from semantic memory based on query"""
        # Both graph_only and vector_only constraints disallow semantic memory
        if self.memory_constraint in ["graph_only", "vector_only"]:
            logger.info(f"Agent {self.agent_id} attempted to use semantic_read but is constrained to {self.memory_constraint}")
            return "Semantic memory is disabled due to memory constraints."
            
        with open(self.semantic_memory_path, 'r') as f:
            memory = json.load(f)
        
        # In a real implementation, this would use semantic similarity search
        # For now, return all entries to the agent to let it do the filtering
        entries = memory["entries"]
        
        # Get the schema for context
        with open(self.schema_path, 'r') as f:
            schemas = json.load(f)
            semantic_schema = schemas.get("semantic_schema", "Default semantic schema")
        
        if not entries:
            return f"Semantic memory is empty. Current schema: {semantic_schema}"
        
        result = f"Semantic Memory (using schema: {semantic_schema}):\n"
        result += "\n".join([f"Entry {e['id']}: {e['content']}" for e in entries])
        
        logger.info(f"Read from semantic memory with query: {query[:30]}...")
        return result
    
    def update_graph_schema(self, new_schema_description):
        """Update graph memory schema"""
        with open(self.schema_path, 'r') as f:
            schemas = json.load(f)
        
        schemas["graph_schema"] = new_schema_description
        
        with open(self.schema_path, 'w') as f:
            json.dump(schemas, f, indent=2)
        
        logger.info(f"Updated graph schema: {new_schema_description[:30]}...")
        return True
    
    def update_vector_schema(self, new_schema_description):
        """Update vector memory schema"""
        with open(self.schema_path, 'r') as f:
            schemas = json.load(f)
        
        schemas["vector_schema"] = new_schema_description
        
        with open(self.schema_path, 'w') as f:
            json.dump(schemas, f, indent=2)
        
        logger.info(f"Updated vector schema: {new_schema_description[:30]}...")
        return True
    
    def update_semantic_schema(self, new_schema_description):
        """Update semantic memory schema"""
        with open(self.schema_path, 'r') as f:
            schemas = json.load(f)
        
        schemas["semantic_schema"] = new_schema_description
        
        with open(self.schema_path, 'w') as f:
            json.dump(schemas, f, indent=2)
        
        logger.info(f"Updated semantic schema: {new_schema_description[:30]}...")
        return True
    
    def clear_all_memory(self):
        """Clear all memory files - used for resetting between games"""
        self._initialize_memory_files()
        
        # Also reset the in-memory GraphMemory and VectorizedMemory
        self.graph_memory = GraphMemory(storage=str(self.graph_pickle_path))
        self.vector_memory = VectorizedMemory(self.autoencoder)
        
        logger.info(f"Cleared all memory for agent {self.agent_id}")
    
    def _get_timestamp(self):
        """Helper function to get current timestamp"""
        import datetime
        return datetime.datetime.now().isoformat() 