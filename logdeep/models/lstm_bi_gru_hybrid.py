import torch
import torch.nn as nn
from logdeep.models.attention import SelfAttention

class HybridLSTMBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, vocab_size, embedding_dim):
        super(HybridLSTMBiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)


        # LSTM layers
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Bi-GRU layers
        self.bi_gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)

        # Self-Attention layer
        self.attention = SelfAttention(hidden_size)

        # Output layer
        self.fc = nn.Linear(hidden_size * 2, vocab_size) # Output layer

    def forward(self, features, device):
        log_seq = features[0].long().to(device)
        batch_size, seq_len = log_seq.size()
        embedded = self.embedding(log_seq)

        # LSTM forward pass
        lstm_out, _ = self.lstm(embedded)

        # Bi-GRU forward pass
        bi_gru_out, _ = self.bi_gru(lstm_out)

        # Self-Attention pass
        attended_output = self.attention(bi_gru_out) # (batch_size, hidden_size * 2)

        # Output layer
        out = self.fc(attended_output)
        return out.view(batch_size, 1, -1) # Reshape to (batch_size, 1, vocab_size)