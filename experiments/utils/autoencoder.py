import torch
from torch import nn
import numpy as np

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Linear(9, 6)
        self.decoder = nn.Linear(6, 9)
    
    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return encoded, decoded
    

def train_encoder(ae, boards, epochs=1000, lr=1e-3):
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    ae.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for board in boards:
            board_tensor = board_to_tensor(board)
            
            opt.zero_grad()
            
            enc_pred, dec_pred = ae(board_tensor)
            
            loss = loss_fn(dec_pred, board_tensor)
            
            loss.backward()
            opt.step()
        
            total_loss += loss

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")


def board_to_tensor(state_str):
    mapping = {'X': 1., 'O': -1., '-': 0.}
    
    board = [mapping[c] for c in state_str]
    
    return torch.tensor(board, dtype=torch.float32).unsqueeze(0)

# Additional utility functions for the experiment setup

def encode_board(ae, state_str):
    """Encode a board state string to its vector embedding"""
    ae.eval()
    with torch.no_grad():
        board_tensor = board_to_tensor(state_str)
        encoded, _ = ae(board_tensor)
        return encoded.squeeze(0).numpy()

def batch_encode_boards(ae, state_strings):
    """Encode multiple board states at once"""
    embeddings = []
    for state in state_strings:
        embeddings.append(encode_board(ae, state))
    return np.array(embeddings)

def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

def find_similar_boards(ae, query_state, board_collection, top_k=3):
    """Find the top_k most similar boards to the query state"""
    query_embedding = encode_board(ae, query_state)
    
    similarities = []
    for i, state in enumerate(board_collection):
        state_embedding = encode_board(ae, state)
        sim = cosine_similarity(query_embedding, state_embedding)
        similarities.append((sim, i))
    
    # Sort by similarity (highest first)
    similarities.sort(reverse=True)
    
    # Return top_k results
    return [(board_collection[idx], sim) for sim, idx in similarities[:top_k]] 