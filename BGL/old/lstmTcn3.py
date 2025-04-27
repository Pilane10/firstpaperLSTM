# -*- coding: utf-8 -*-
import platform
import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sys.path.append('../../')

from logdeep.tools.predict_lstmTcn import Predicter
from logdeep.tools.train_lstmTcn import EnhancedTrainer
from logdeep.tools.utils import *
from logdeep.dataset.vocab import Vocab


# ====================== Enhanced Model Definition ======================
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
        energy = torch.tanh(self.attn(hidden))
        scores = self.v(energy).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)
        attn_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), hidden).squeeze(1)
        return context, attn_weights


class EnhancedLstmTcn(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, vocab_size, embedding_dim, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Enhanced TCN with multiple dilated layers
        self.tcn = nn.Sequential(
            TCNBlock(embedding_dim, embedding_dim, dilation=1, dropout=dropout),
            TCNBlock(embedding_dim, embedding_dim, dilation=2, dropout=dropout),
            TCNBlock(embedding_dim, embedding_dim, dilation=4, dropout=dropout)
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism
        self.attention = Attention(hidden_size * 2)

        # Output heads
        self.next_token_head = nn.Linear(hidden_size * 2, vocab_size)
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x, lengths=None, return_attention=False):
        # Embedding
        x = self.embedding(x)

        # TCN processing
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)

        # Handle variable lengths with packing
        if lengths is not None:
            x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # LSTM processing
        lstm_out, _ = self.lstm(x)

        if lengths is not None:
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        # Attention pooling
        context, attn_weights = self.attention(lstm_out)

        # Outputs
        next_token_pred = self.next_token_head(lstm_out[:, -1, :])
        anomaly_score = self.anomaly_head(context)

        if return_attention:
            return next_token_pred, anomaly_score, attn_weights
        return next_token_pred, anomaly_score


# ====================== Main Script ======================
output_dir = "../../output/bgl/"

# Config Parameters
options = {
    'output_dir': output_dir,
    'train_vocab': output_dir + 'train',
    'vocab_path': output_dir + "vocab.pkl",
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'sample': "sliding_window",
    'window_size': 20,
    'train_ratio': 1,
    'valid_ratio': 0.1,
    'test_ratio': 1,
    'min_len': 10,
    'is_logkey': True,
    'is_time': False,
    'sequentials': True,  # Set to options["is_logkey"] later
    'quantitatives': False,
    'semantics': False,
    'parameters': False,  # Set to options["is_time"] later
    'input_size': 1,
    'hidden_size': 64,
    'num_layers': 4,
    'embedding_dim': 100,
    'vocab_size': 200,
    'dropout': 0.3,
    'batch_size': 512,
    'accumulation_step': 2,
    'optimizer': 'adam',
    'lr': 2e-3,
    'max_epoch': 5,
    'n_epochs_stop': 10,
    'weight_decay': 1e-5,
    'lr_step': (5, 10),
    'lr_decay_ratio': 0.5,
    'pos_weight': 5.0,
    'resume_path': None,
    'model_name': "enhanced_lstm_tcn",
    'num_candidates': 9,
    'threshold': None,
    'gaussian_mean': 0,
    'gaussian_std': 0,
    'num_outputs': 2
}

# Set dependent options
options['sequentials'] = options["is_logkey"]
options['parameters'] = options["is_time"]
options['feature_num'] = sum([options['sequentials'], options['quantitatives'],
                              options['semantics'], options['parameters']])
options['num_classes'] = options["vocab_size"]
#options['save_dir'] = os.path.join(options["output_dir"], options["model_name"])
options['save_dir'] = options["output_dir"] + "enhanced_lstm_tcn/"
#options['model_path'] = os.path.join(options["save_dir"], "bestloss.pth")
options['model_path'] = options["save_dir"] + "bestloss.pth"

print("Features logkey:{} time: {}".format(options["is_logkey"], options["is_time"]))
print("Device:", options['device'])

seed_everything(seed=1234)

# Initialize enhanced model
Model = EnhancedLstmTcn(
    input_size=options['input_size'],
    hidden_size=options['hidden_size'],
    num_layers=options['num_layers'],
    vocab_size=options["vocab_size"],
    embedding_dim=options["embedding_dim"],
    dropout=options['dropout']
)


def train():
    trainer = EnhancedTrainer(Model, options)
    trainer.start_train()


def predict():
    predicter = Predicter(Model, options)
    predicter.predict_unsupervised()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(mode='train')

    predict_parser = subparsers.add_parser('predict')
    predict_parser.set_defaults(mode='predict')

    vocab_parser = subparsers.add_parser('vocab')
    vocab_parser.set_defaults(mode='vocab')

    args = parser.parse_args()
    print("arguments", args)

    if args.mode == 'train':
        train()
    elif args.mode == 'predict':
        predict()
    elif args.mode == 'vocab':
        with open(options["train_vocab"], 'r') as f:
            logs = f.readlines()
        vocab = Vocab(logs)
        print("vocab_size", len(vocab))
        vocab.save_vocab(options["vocab_path"])