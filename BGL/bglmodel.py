# -*- coding: utf-8 -*-
import platform
import argparse
import sys
import os

# --- Replace this with the actual path to your project_root directory ---
project_root = "/home/pilane/PycharmProjects/pilanelogbert/"

sys.path.append(os.path.join(project_root, 'logdeep')) # Append the logdeep directory to access its modules

from logdeep.models.lstm import Deeplog
from logdeep.tools.predict import Predicter
from logdeep.tools.train import Trainer
from logdeep.tools.utils import seed_everything
from logdeep.dataset.vocab import Vocab

import torch

output_dir = os.path.join(project_root, "output/bgl/")

# Config Parameters
options = dict()
options['output_dir'] = output_dir
options['train_vocab'] = os.path.join(output_dir, 'train')
options["vocab_path"] = os.path.join(output_dir, "vocab.pkl")

options['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

# Sample
options['sample'] = "sliding_window"
options['window_size'] = 20  # if fix_window
options['train_ratio'] = 1.0 # Assuming your 'train' file contains all training data
options['valid_ratio'] = 0.1
options['test_ratio'] = 1.0
options["min_len"] = 10

options["is_logkey"] = True
options["is_time"] = False

# Features
options['sequentials'] = options["is_logkey"]
options['quantitatives'] = False
options['semantics'] = False
options['parameters'] = options["is_time"]
options['feature_num'] = sum(
    [options['sequentials'], options['quantitatives'], options['semantics'], options['parameters']])

# Model
options['input_size'] = 1
options['hidden_size'] = 64
options['num_layers'] = 2
options["embedding_dim"] = 50
options["vocab_size"] = 200 # Initial vocab size, will be updated based on data
options['num_classes'] = options["vocab_size"]
# Train
options['batch_size'] = 128
options['accumulation_step'] = 1

options['optimizer'] = 'adam'
options['lr'] = 0.01
options['max_epoch'] = 15
options["n_epochs_stop"] = 10
options['lr_step'] = (options['max_epoch'] - 20, options['max_epoch'])
options['lr_decay_ratio'] = 0.1

options['resume_path'] = None
options['model_name'] = "deeplog"
options['save_dir'] = os.path.join(options["output_dir"], "deeplog/")
os.makedirs(options['save_dir'], exist_ok=True)

# Predict
options['model_path'] = os.path.join(options["save_dir"], "bestloss.pth")
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


def train():
    trainer = Trainer(Model, options)
    trainer.start_train()


def predict():
    predicter = Predicter(Model, options)
    predicter.predict_unsupervised()


def vocab_generation():
    with open(options["train_vocab"], 'r') as f:
        logs = f.readlines()
    vocab = Vocab(logs)
    print("vocab_size", len(vocab))
    vocab.save_vocab(options["vocab_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(mode='train')

    predict_parser = subparsers.add_parser('predict')
    predict_parser.set_defaults(mode='predict')
    predict_parser.add_argument('--mean', type=float, default=0, help='error gaussian distribution mean')
    predict_parser.add_argument('--std', type=float, default=0, help='error gaussian distribution std')

    vocab_parser = subparsers.add_parser('vocab')
    vocab_parser.set_defaults(mode='vocab')

    import sys
    if not "ipykernel" in sys.modules:
        args = parser.parse_args()
        print("arguments", args)

        if args.mode == 'train':
            train()
        elif args.mode == 'predict':
            predict()
        elif args.mode == 'vocab':
            vocab_generation()
        else:
            print("Please specify a mode: 'train', 'predict', or 'vocab'")
    else:
        print("Running in Jupyter. Call train(), predict(), or vocab_generation() functions directly in separate cells.")