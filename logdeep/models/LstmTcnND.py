import torch
import torch.nn as nn
import torch.nn.functional as F

# Temporal Convolutional Network Block
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2, use_layer_norm=True):
        super(TCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)

        # Removed BatchNorm1d as requested by omitting attention-specific normalization
        self.norm1 = nn.LayerNorm(out_channels) if use_layer_norm else None
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        # Removed BatchNorm1d
        self.norm2 = nn.LayerNorm(out_channels) if use_layer_norm else None
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection (to help gradients flow)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        # Save input for residual connection
        identity = x

        out = self.conv1(x)
        # Layer norm can be added here if desired per convolution block,
        # but typically LayerNorm is applied after attention/RNN blocks or globally.

        # First convolution block
        out = self.conv1(x)
        # Permute for LayerNorm (which expects channels last)
        if self.norm1 is not None:
            out = out.permute(0, 2, 1)  # [batch, seq_len, channels]
            out = self.norm1(out)
            out = out.permute(0, 2, 1)  # back to [batch, channels, seq_len]

        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        # Layer norm here too if per-block

        # Second convolution block
        out = self.conv2(out)
        if self.norm2 is not None:
            out = out.permute(0, 2, 1)  # [batch, seq_len, channels]
            out = self.norm2(out)
            out = out.permute(0, 2, 1)  # back to [batch, channels, seq_len]

        out = self.relu2(out)
        out = self.dropout2(out)

        # Apply residual connection
        if self.downsample is not None:
            identity = self.downsample(identity)

        # Ensure dimensions match for the addition
        # TCN's padding handles sequence length mismatch due to convolution,
        # but the residual connection needs channel dimensions to match.
        return F.relu(out + identity)


# Hybrid LSTM + TCN Model without Attention, with Normalization and Dropout
class LstmTcnWithoutAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, vocab_size, embedding_dim,
                 dropout=0.2, use_layer_norm=True):
        # 'heads' parameter is removed as attention is removed
        super(LstmTcnWithoutAttention, self).__init__()

        # Embedding layer to convert log keys to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # TCN block with dropout
        # Use embedding_dim for both input and output of TCN for simpler integration with LSTM
        self.tcn = TCNBlock(in_channels=embedding_dim, out_channels=embedding_dim, dropout=dropout, use_layer_norm=use_layer_norm)

        # LSTM layer with dropout
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)

        # Final output layer to map LSTM hidden state to vocab size (log key prediction)
        self.fc = nn.Linear(hidden_size, vocab_size)

        # Attention layer is removed
        # self.attention = nn.MultiheadAttention(...) # Removed

        # Layer normalization (if enabled)
        # Normalization is typically applied to the output of LSTM/RNN before the final linear layer
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)  # Use hidden_size for normalization
        else:
            self.layer_norm = None

        # Dropout after normalization or directly on LSTM output if no normalization
        self.final_dropout = nn.Dropout(dropout) # Added an extra dropout layer after potential layer norm

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
        # lstm_out is [batch_size, seq_len, hidden_size]
        lstm_out, _ = self.lstm(x)

        # Attention mechanism is removed

        # Apply layer normalization if enabled to the LSTM output
        if self.layer_norm:
            lstm_out = self.layer_norm(lstm_out)

        # Apply dropout to the LSTM output (after potential normalization)
        lstm_out = self.final_dropout(lstm_out)

        # Take the output corresponding to the last time step for each sequence
        # [batch_size, hidden_size]
        # We use the last state of the output sequence from the LSTM,
        # as attention is removed and we are typically predicting the next log key
        # based on the sequence history captured by the final state.
        final_state = lstm_out[:, -1, :]

        # Output logits: [batch_size, vocab_size]
        logits = self.fc(final_state)

        return logits

# Example Usage:
# input_size (not directly used in this version as embedding handles input size),
# hidden_size (LSTM output size), num_layers (LSTM layers), vocab_size (number of unique log keys),
# embedding_dim (size of log key embeddings), dropout rate, use_layer_norm flag.

# vocab_size = 1000
# embedding_dim = 128
# hidden_size = 256
# num_layers = 2
# dropout_rate = 0.3
# use_norm = True

# model = LstmTcnWithoutAttention(input_size=None, # input_size is implicit via embedding_dim
#                                 hidden_size=hidden_size,
#                                 num_layers=num_layers,
#                                 vocab_size=vocab_size,
#                                 embedding_dim=embedding_dim,
#                                 dropout=dropout_rate,
#                                 use_layer_norm=use_norm)

# print(model)

# Example forward pass (dummy data)
# batch_size = 32
# seq_len = 50
# dummy_input = torch.randint(1, vocab_size, (batch_size, seq_len)) # Log key indices

# output_logits = model(dummy_input)
# print("Output logits shape:", output_logits.shape) # Should be [batch_size, vocab_size]