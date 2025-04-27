# -*- coding: utf-8 -*-
import platform
import argparse
import sys
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import pickle

sys.path.append('../')

from logdeep.models.lstm import *
from logdeep.tools.predict import Predicter
from logdeep.tools.train import Trainer
from logdeep.tools.utils import *
from logdeep.dataset.vocab import Vocab

output_dir = "../output/bgl/"

# Config Parameters
options = dict()
options["vocab_path"] = output_dir + "vocab.pkl"

options['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

# Sample
options['sample'] = "sliding_window"
options['window_size'] = 20  # if fix_window
options['train_ratio'] = 1
options['valid_ratio'] = 0.1
options['test_ratio'] = 1
options["min_len"] = 10

options["is_logkey"] = True
options["is_time"] = False

# Features
options['sequentials'] = options["is_logkey"]
options['quantitatives'] = False
options['semantics'] = False
options['parameters'] = options["is_time"]
options['feature_num'] = sum(
    [options['sequentials'], options['quantitatives'], options['semantics'], options['parameters']]
)

# Model
options['input_size'] = 1
options['hidden_size'] = 64
options['num_layers'] = 2
options["embedding_dim"] = 50
options["vocab_size"] = 200
options['num_classes'] = options["vocab_size"]

# Train
options['batch_size'] = 128
options['accumulation_step'] = 1

options['optimizer'] = 'adam'
options['lr'] = 0.01
options['max_epoch'] = 1
options["n_epochs_stop"] = 10
options['lr_step'] = (options['max_epoch'] - 20, options['max_epoch'])
options['lr_decay_ratio'] = 0.1

options['resume_path'] = None
options['model_name'] = "deeplog"

# Predict
options['num_candidates'] = 9
options["threshold"] = None
options["gaussian_mean"] = 0
options["gaussian_std"] = 0
options["num_outputs"] = 1


print("Features logkey:{} time: {}".format(options["is_logkey"], options["is_time"]))
print("Device:", options['device'])

seed_everything(seed=1234)

Model = Deeplog(input_size=options['input_size'],
                hidden_size=options['hidden_size'],
                num_layers=options['num_layers'],
                vocab_size=options["vocab_size"],
                embedding_dim=options["embedding_dim"])


# Custom Dataset Class
class LogDataset(Dataset):
    def __init__(self, file_path, vocab):
        """
        Dataset for loading log sequences.

        Args:
            file_path (str): Path to the file containing sequences.
            vocab (dict): Vocabulary mapping Event IDs to numerical IDs.
        """
        self.sequences = []
        self.labels = []

        with open(file_path, "r") as f:
            for line in f:
                event_ids = line.strip().split()
                num_sequence = [vocab.get(event_id, 1) for event_id in event_ids]  # Use 1 for UNK
                self.sequences.append(num_sequence)
                self.labels.append(0)  # Assuming normal logs as 0 (update if labels are provided)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# Custom Collate Function
def custom_collate_fn(batch):
    """
    Custom collate function to handle padding of sequences in a batch.

    Args:
        batch (list of tuples): Each element in the batch is a tuple (sequence, label).

    Returns:
        tuple: Padded sequences and corresponding labels.
    """
    sequences, labels = zip(*batch)  # Separate sequences and labels
    sequences = [torch.tensor(seq) for seq in sequences]  # Convert to tensors
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)  # Pad sequences
    labels = torch.tensor(labels)  # Convert labels to tensor
    return padded_sequences, labels


def train_all_folds():
    """
    Train the model iteratively for all folds.
    """
    fold_dirs = [os.path.join(output_dir, fold) for fold in os.listdir(output_dir) if fold.startswith("fold_")]
    fold_dirs.sort()  # Ensure consistent order (fold_1, fold_2, ...)

    # Load vocabulary
    with open(options["vocab_path"], "rb") as f:
        vocab = pickle.load(f)

    for fold_dir in fold_dirs:
        print(f"\nStarting training for {fold_dir}...\n")

        # Update paths for the current fold
        train_file = os.path.join(fold_dir, 'train')  # Ensure correct path
        valid_file = os.path.join(fold_dir, 'tuning')  # Ensure correct path
        test_file = os.path.join(fold_dir, 'eval')  # Ensure correct path

        # Create a unique directory for this fold's results
        fold_output_dir = os.path.join(fold_dir, "deeplog")
        os.makedirs(fold_output_dir, exist_ok=True)

        # Update options to point to the correct files and output directory for the current fold
        options['train_data_path'] = train_file
        options['valid_data_path'] = valid_file
        options['test_data_path'] = test_file
        options['save_dir'] = fold_output_dir  # Set unique directory for saving results
        options['output_dir'] = fold_dir  # Set the current fold's directory as output_dir

        # Create Trainer for the current fold
        trainer = Trainer(Model, options)

        # Start training
        trainer.start_train()

        print(f"\nCompleted training for {fold_dir}. Results saved to {fold_output_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_folds', action='store_true', help='Train iteratively for all folds')

    args = parser.parse_args()

    if args.all_folds:
        train_all_folds()
    else:
        print("Please specify --all_folds to iterate through all folds.")