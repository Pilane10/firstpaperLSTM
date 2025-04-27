import torch
import torch.nn as nn
import torch.nn.functional as F

# Temporal Convolutional Network Block
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super(TCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection (to help gradients flow)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)

# Hybrid LSTM + TCN Model
class LstmTcn(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, vocab_size, embedding_dim):
        super(LstmTcn, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer to convert log keys to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # TCN block: (in_channels, out_channels)
        self.tcn = TCNBlock(in_channels=embedding_dim, out_channels=embedding_dim)

        # LSTM: input_size must match TCN output channels
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        # Final output layer to map LSTM hidden state to vocab size (log key prediction)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # x: [batch_size, seq_len]

        # Embedding lookup: [batch_size, seq_len, embedding_dim]
        x = self.embedding(x)

        # TCN expects input shape: [batch_size, embedding_dim, seq_len]
        x = x.permute(0, 2, 1)

        # Pass through TCN: output shape [batch_size, embedding_dim, seq_len]
        x = self.tcn(x)

        # Permute back for LSTM: [batch_size, seq_len, embedding_dim]
        x = x.permute(0, 2, 1)

        # LSTM expects (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)

        # Take final LSTM output state for each sequence: [batch_size, hidden_size]
        final_state = lstm_out[:, -1, :]

        # Output logits: [batch_size, vocab_size]
        logits = self.fc(final_state)

        return logits
