import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.projection = nn.Linear(hidden_size * 2, 1) # Assuming Bi-GRU outputs double the hidden size

    def forward(self, encoder_outputs):
        # encoder_outputs: (batch_size, seq_len, hidden_size * 2)
        energy = self.projection(encoder_outputs) # (batch_size, seq_len, 1)
        weights = torch.softmax(energy.squeeze(-1), dim=-1) # (batch_size, seq_len)
        # weights: (batch_size, seq_len)
        # encoder_outputs: (batch_size, seq_len, hidden_size * 2)
        attended = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1) # (batch_size, hidden_size * 2)
        return attended