# train_hybrid_without_attention.py
import sys
sys.path.append('../')
from logdeep.models.lstm_bi_gru_hybrid import HybridLSTMBiGRU
from logdeep.tools.train_hybrid_Nsa import Trainer # Assuming you saved the without-attention trainer here
import pickle
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Hybrid LSTM-BiGRU without Attention')
    parser.add_argument('--vocab_path', type=str, default='output/bgl/vocab.pkl', help='Path to vocabulary')
    parser.add_argument('--output_dir', type=str, default='output/bgl/', help='Path to output directory')
    parser.add_argument('--save_dir', type=str, default='output/bgl/hybrid_without_attention/', help='Path to save model')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size of LSTM and GRU')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM/GRU layers')
    parser.add_argument('--window_size', type=int, default=10, help='Window size for sliding window')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--max_epoch', type=int, default=10, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    # Add other necessary hyperparameters here
    args = parser.parse_args()
    options = vars(args)
    options['sequentials'] = True
    options['quantitatives'] = False
    options['semantics'] = False
    options['parameters'] = False
    options['sample'] = 'sliding_window'
    options['train_ratio'] = 0.8
    options['valid_ratio'] = 0.2
    options['is_logkey'] = True
    options['is_time'] = False
    options['min_len'] = 1

    with open(options['vocab_path'], 'rb') as f:
        vocab = pickle.load(f)
        vocab_size = len(vocab)

    model = HybridLSTMBiGRU(input_size=options['embedding_dim'],
                             hidden_size=options['hidden_size'],
                             num_layers=options['num_layers'],
                             vocab_size=vocab_size,
                             embedding_dim=options['embedding_dim'],
                             use_attention=False)

    trainer = Trainer(model, options)
    trainer.start_train()