import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch

class VectorizedMemory:
    def __init__(self, encoder_model):
        self.encoder_model = encoder_model.eval()
        self.memory_vectors = []
        self.memory_metadata = []
        self.nn_model = None  # Nearest neighbor model

    def encode(self, board):
        """Encode a board (string or flattened) into a latent vector."""
        board_tensor = self.board_to_tensor(board)
        with torch.no_grad():
            _, encoded = self.encoder_model(board_tensor)
        return encoded.squeeze(0).cpu().numpy()

    def board_to_tensor(self, board):
        """Convert board string like 'XOXOXO   ' into tensor."""
        mapping = {'X': 1.0, 'O': -1.0, ' ': 0.0, '-': 0.0}
        board_list = [mapping[c] for c in board]
        return torch.tensor(board_list, dtype=torch.float32).unsqueeze(0)

    def store(self, board, metadata=None):
        """Store encoded board state."""
        vec = self.encode(board)
        self.memory_vectors.append(vec)
        self.memory_metadata.append(metadata)
        self._rebuild_nn()

    def retrieve(self, board, k=3):
        """Retrieve k nearest memories."""
        if not self.memory_vectors or not self.nn_model:
            return []
        vec = self.encode(board)
        distances, indices = self.nn_model.kneighbors(np.array([vec]), n_neighbors=min(k, len(self.memory_vectors)))
        results = [(self.memory_metadata[i]) for i in indices[0]]
        return results

    def _rebuild_nn(self):
        """Rebuild nearest neighbor search index."""
        if len(self.memory_vectors) > 0:
            self.nn_model = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(np.array(self.memory_vectors)) 