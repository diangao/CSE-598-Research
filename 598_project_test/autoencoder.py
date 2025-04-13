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


# ae = AutoEncoder()

# boards = [
#   "XOXOXO---",
#   "XOX-OX--O",
#   "XO-XOX--O",
#   "OX-XO--XO",
#   "XO--XOXO-",
#   "OXXOO-X--",
#   "XOXO--X--",
#   "XO-OXOX--",
#   "X-OXOXO--",
#   "OXO-XOX--",
#   "XOXOOX---",
#   "OXOXOXO--",
#   "XOX-OX-O-",
#   "OXXO--X-O",
#   "OXOXO-X--",
#   "XO-OX-OXX",
#   "XOXOX--O-",
#   "O-XXOXO--",
#   "OX-OXOX--",
#   "XOXXO--O-"
# ]

# ae.load_state_dict(torch.load('balls.model', weights_only=True))

# ae.eval()