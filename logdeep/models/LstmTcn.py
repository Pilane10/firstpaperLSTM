import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ====================== Model Components ======================
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.relu(self.conv2(out))
        out = self.dropout(out)
        return F.relu(out + residual)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, mask=None):
        # hidden: [batch, seq_len, hidden_size]
        energy = torch.tanh(self.attn(hidden))  # [batch, seq_len, hidden_size]
        scores = self.v(energy).squeeze(2)  # [batch, seq_len]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)

        attn_weights = F.softmax(scores, dim=1)  # [batch, seq_len]
        context = torch.bmm(attn_weights.unsqueeze(1), hidden).squeeze(1)  # [batch, hidden_size]
        return context, attn_weights


# ====================== Main Model ======================
class LogAnomalyDetector(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, tcn_channels=64,
                 lstm_hidden=128, num_layers=2, dropout=0.3):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # TCN Network (3 blocks with increasing dilation)
        self.tcn = nn.Sequential(
            TCNBlock(embedding_dim, tcn_channels, dilation=1, dropout=dropout),
            TCNBlock(tcn_channels, tcn_channels, dilation=2, dropout=dropout),
            TCNBlock(tcn_channels, tcn_channels, dilation=4, dropout=dropout)
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=tcn_channels,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism
        self.attention = Attention(lstm_hidden * 2)

        # Anomaly detection head
        self.anomaly_head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 1),
            nn.Sigmoid()
        )

        # Next-token prediction head (auxiliary task)
        self.next_token_head = nn.Linear(lstm_hidden * 2, vocab_size)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x, lengths=None):
        """
        Args:
            x: [batch_size, seq_len] - input log sequences
            lengths: [batch_size] - actual lengths of sequences
        Returns:
            anomaly_score: [batch_size, 1] - probability of anomaly
            next_token_pred: [batch_size, vocab_size] - predicted next log key
        """
        # Embedding
        x = self.embedding(x)  # [batch, seq_len, emb_dim]

        # TCN processing
        x = x.permute(0, 2, 1)  # [batch, emb_dim, seq_len]
        x = self.tcn(x)  # [batch, tcn_channels, seq_len]
        x = x.permute(0, 2, 1)  # [batch, seq_len, tcn_channels]

        # Handle variable lengths with packing
        if lengths is not None:
            x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # LSTM processing
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, lstm_hidden*2]

        if lengths is not None:
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        # Attention pooling
        context, attn_weights = self.attention(lstm_out)  # [batch, lstm_hidden*2]

        # Anomaly detection
        anomaly_score = self.anomaly_head(context)  # [batch, 1]

        # Next-token prediction (auxiliary task)
        next_token_pred = self.next_token_head(lstm_out[:, -1, :])  # [batch, vocab_size]

        return anomaly_score, next_token_pred


# ====================== Training Utilities ======================
class WeightedLoss(nn.Module):
    """Handles class imbalance by weighting anomalies higher"""

    def __init__(self, pos_weight=5.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        weight = torch.where(target == 1, self.pos_weight, 1.0)
        return (weight * bce_loss).mean()


def train_step(model, batch, optimizer, criterion):
    """Single training step"""
    model.train()
    x, y_anomaly, y_next, lengths = batch

    # Forward pass
    anomaly_scores, next_preds = model(x, lengths)

    # Calculate losses
    anomaly_loss = criterion(anomaly_scores.squeeze(), y_anomaly.float())
    next_loss = F.cross_entropy(next_preds, y_next)
    total_loss = anomaly_loss + 0.3 * next_loss  # Weighted sum

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
    optimizer.step()

    return total_loss.item()


# ====================== Example Usage ======================
if __name__ == "__main__":
    # Hyperparameters
    config = {
        'vocab_size': 100,
        'embedding_dim': 64,
        'tcn_channels': 64,
        'lstm_hidden': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'lr': 0.001,
        'pos_weight': 5.0  # Higher weight for anomalies
    }

    # Initialize model
    model = LogAnomalyDetector(**config)

    # Example inputs
    batch_size, seq_len = 32, 100
    x = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    lengths = torch.randint(50, seq_len, (batch_size,))  # Variable lengths
    y_anomaly = torch.randint(0, 2, (batch_size,))  # Binary anomaly labels
    y_next = torch.randint(0, config['vocab_size'], (batch_size,))  # Next token

    # Forward pass
    anomaly_scores, next_preds = model(x, lengths)
    print(f"Anomaly scores shape: {anomaly_scores.shape}")
    print(f"Next preds shape: {next_preds.shape}")