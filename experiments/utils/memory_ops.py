import json
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, agent_id, memory_base_dir="experiments/agents"):
        """Initialize memory manager for an agent"""
        self.agent_id = agent_id
        self.memory_dir = Path(f"{memory_base_dir}/{agent_id}/memory")
        
        # Create memory files if they don't exist
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize memory files with empty data
        self.graph_memory_path = self.memory_dir / "graph_memory.json"
        self.vector_memory_path = self.memory_dir / "vector_memory.json"
        self.semantic_memory_path = self.memory_dir / "semantic_memory.json"
        self.schema_path = self.memory_dir / "schemas.json"
        
        # Initialize memory files if they don't exist
        self._initialize_memory_files()
        
        logger.info(f"Memory manager initialized for agent {agent_id}")
    
    def _initialize_memory_files(self):
        """Initialize memory files with empty data if they don't exist"""
        # Initialize graph memory
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
        """Store content in graph memory"""
        with open(self.graph_memory_path, 'r') as f:
            memory = json.load(f)
        
        # Add to entries
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
        """Read from graph memory based on query"""
        with open(self.graph_memory_path, 'r') as f:
            memory = json.load(f)
        
        # In a real implementation, this would use more sophisticated matching
        # For now, return all entries to the agent to let it do the filtering
        entries = memory["entries"]
        
        # Get the schema for context
        with open(self.schema_path, 'r') as f:
            schemas = json.load(f)
            graph_schema = schemas.get("graph_schema", "Default graph schema")
        
        if not entries:
            return f"Graph memory is empty. Current schema: {graph_schema}"
        
        result = f"Graph Memory (using schema: {graph_schema}):\n"
        result += "\n".join([f"Entry {e['id']}: {e['content']}" for e in entries])
        
        logger.info(f"Read from graph memory with query: {query[:30]}...")
        return result
    
    def vector_store(self, content):
        """Store content in vector memory"""
        with open(self.vector_memory_path, 'r') as f:
            memory = json.load(f)
        
        # Add to entries
        memory["entries"].append({
            "id": len(memory["entries"]),
            "content": content,
            "timestamp": self._get_timestamp()
        })
        
        with open(self.vector_memory_path, 'w') as f:
            json.dump(memory, f, indent=2)
        
        logger.info(f"Stored in vector memory: {content[:30]}...")
        return True
    
    def vector_read(self, query):
        """Read from vector memory based on query"""
        with open(self.vector_memory_path, 'r') as f:
            memory = json.load(f)
        
        # In a real implementation, this would use vector similarity search
        # For now, return all entries to the agent to let it do the filtering
        entries = memory["entries"]
        
        # Get the schema for context
        with open(self.schema_path, 'r') as f:
            schemas = json.load(f)
            vector_schema = schemas.get("vector_schema", "Default vector schema")
        
        if not entries:
            return f"Vector memory is empty. Current schema: {vector_schema}"
        
        result = f"Vector Memory (using schema: {vector_schema}):\n"
        result += "\n".join([f"Entry {e['id']}: {e['content']}" for e in entries])
        
        logger.info(f"Read from vector memory with query: {query[:30]}...")
        return result
    
    def semantic_store(self, content):
        """Store content in semantic memory"""
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
        logger.info(f"Cleared all memory for agent {self.agent_id}")
    
    def _get_timestamp(self):
        """Helper function to get current timestamp"""
        import datetime
        return datetime.datetime.now().isoformat() 