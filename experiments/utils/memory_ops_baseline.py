import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Memory-less version of MemoryManager - maintains the same API but doesn't perform actual memory operations
    Used for baseline experiments
    """
    
    def __init__(self, agent_id, memory_base_dir="experiments/agents", memory_constraint="none", **kwargs):
        """Initialize memory-less manager"""
        self.agent_id = agent_id
        self.memory_constraint = "baseline"  # Force set to baseline
        self.memory_dir = Path(f"{memory_base_dir}/{agent_id}/memory")
        
        # Create memory directory (for compatibility only)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized BASELINE (NO-MEMORY) manager for agent {agent_id}")
    
    # Graph memory operations
    def graph_store(self, content):
        """Simulate storing content to graph memory"""
        logger.debug(f"[BASELINE] Ignored graph_store: {content[:30]}...")
        return True
    
    def graph_read(self, query):
        """Simulate reading from graph memory"""
        logger.debug(f"[BASELINE] Ignored graph_read: {query[:30]}...")
        return "This is a baseline agent without memory capabilities."
    
    # Vector memory operations
    def vector_store(self, content, board_state=None):
        """Simulate storing content to vector memory"""
        logger.debug(f"[BASELINE] Ignored vector_store: {content[:30]}...")
        return True
    
    def vector_read(self, query):
        """Simulate reading from vector memory"""
        logger.debug(f"[BASELINE] Ignored vector_read: {query[:30]}...")
        return "This is a baseline agent without memory capabilities."
    
    # Semantic memory operations
    def semantic_store(self, content):
        """Simulate storing content to semantic memory"""
        logger.debug(f"[BASELINE] Ignored semantic_store: {content[:30]}...")
        return True
    
    def semantic_read(self, query):
        """Simulate reading from semantic memory"""
        logger.debug(f"[BASELINE] Ignored semantic_read: {query[:30]}...")
        return "This is a baseline agent without memory capabilities."
    
    # Schema update operations
    def update_graph_schema(self, new_schema_description):
        """Simulate updating graph memory schema"""
        logger.debug(f"[BASELINE] Ignored update_graph_schema")
        return True
    
    def update_vector_schema(self, new_schema_description):
        """Simulate updating vector memory schema"""
        logger.debug(f"[BASELINE] Ignored update_vector_schema")
        return True
    
    def update_semantic_schema(self, new_schema_description):
        """Simulate updating semantic memory schema"""
        logger.debug(f"[BASELINE] Ignored update_semantic_schema")
        return True
    
    # Clear memory - no operation
    def clear_all_memory(self):
        """Simulate clearing all memory"""
        logger.debug("[BASELINE] No memory to clear")
        return True
    
    # Compatibility methods
    def train_autoencoder_on_memory(self, epochs=500):
        """Simulate training autoencoder"""
        logger.debug("[BASELINE] No autoencoder to train")
        return True
    
    def _get_timestamp(self):
        """Return current timestamp - keep this method to ensure API compatibility"""
        from datetime import datetime
        return datetime.now().isoformat() 