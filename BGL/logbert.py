import sys

sys.path.append("../")
import argparse
import os
import pandas as pd
from sklearn.model_selection import KFold
from bert_pytorch.dataset import WordVocab
from bert_pytorch import Predictor, Trainer
from logdeep.tools.utils import *

# Hyperparameters and options
options = dict()
options['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
options["output_dir"] = "../output/bgl/"
options["model_dir"] = options["output_dir"] + "bert/"
options["model_path"] = options["model_dir"] + "best_bert.pth"
options["train_vocab"] = options['output_dir'] + 'train'
options["vocab_path"] = options["output_dir"] + "vocab.pkl"

options["window_size"] = 128
options["adaptive_window"] = True
options["seq_len"] = 512
options["max_len"] = 512  # for position embedding
options["min_len"] = 10

options["mask_ratio"] = 0.5

options["train_ratio"] = 1
options["valid_ratio"] = 0.1
options["test_ratio"] = 1

# Features
options["is_logkey"] = True
options["is_time"] = False

options["hypersphere_loss"] = True
options["hypersphere_loss_test"] = False

options["scale"] = None
options["scale_path"] = options["model_dir"] + "scale.pkl"

# Model
options["hidden"] = 256  # embedding size
options["layers"] = 4
options["attn_heads"] = 4

options["epochs"] = 21
options["n_epochs_stop"] = 10
options["batch_size"] = 32

options["corpus_lines"] = None
options["on_memory"] = True
options["num_workers"] = 5
options["lr"] = 1e-3
options["adam_beta1"] = 0.9
options["adam_beta2"] = 0.999
options["adam_weight_decay"] = 0.00
options["with_cuda"] = True
options["cuda_devices"] = None
options["log_freq"] = None

# Predict
options["num_candidates"] = 15
options["gaussian_mean"] = 0
options["gaussian_std"] = 1

seed_everything(seed=1234)
print("device", options["device"])
print("features logkey:{} time: {}".format(options["is_logkey"], options["is_time"]))
print("mask ratio", options["mask_ratio"])


import os

def train_k_fold(options, n_splits=5):
    """
    Function to train the LogBERT model using K-Fold Cross-Validation.
    """
    # Ensure the model directory exists
    os.makedirs(options["model_dir"], exist_ok=True)

    # Load pre-split folds from the output directory
    for fold_idx in range(1, n_splits + 1):
        fold_dir = os.path.join(options["output_dir"], f"fold_{fold_idx}")
        train_file = os.path.join(fold_dir, "train_normal")  # No .csv extension
        test_normal_file = os.path.join(fold_dir, "test_normal")  # No .csv extension
        test_abnormal_file = os.path.join(fold_dir, "test_abnormal")  # No .csv extension

        if not os.path.exists(train_file) or not os.path.exists(test_normal_file) or not os.path.exists(test_abnormal_file):
            raise FileNotFoundError(f"Required files not found in {fold_dir}. Ensure data_process.py has been run correctly.")

        # Load data while skipping problematic rows
        train_data = pd.read_csv(train_file, header=None, names=["EventId"], sep=" ", error_bad_lines=False).dropna()
        val_data_normal = pd.read_csv(test_normal_file, header=None, names=["EventId"], sep=" ", error_bad_lines=False).dropna()
        val_data_anomaly = pd.read_csv(test_abnormal_file, header=None, names=["EventId"], sep=" ", error_bad_lines=False).dropna()

        # Combine validation data (normal + anomaly logs)
        val_data = pd.concat([val_data_normal, val_data_anomaly], ignore_index=True)

        print(f"Training Fold {fold_idx}...")

        # Initialize Trainer and start training
        trainer = Trainer(options)
        trainer.train()  # Call the train method without arguments


def create_vocab(options):
    """
    Function to create the vocabulary for the LogBERT model.

    Combines all `train_normal` files from K-Fold splits to create the vocabulary.
    """
    vocab_lines = []
    for fold_idx in range(1, 6):  # Assuming 5 folds
        fold_dir = os.path.join(options["output_dir"], f"fold_{fold_idx}")
        train_file = os.path.join(fold_dir, "train_normal")
        if os.path.exists(train_file):
            with open(train_file, 'r') as f:
                vocab_lines.extend(f.readlines())
        else:
            print(f"Warning: Train file not found for Fold {fold_idx}, skipping.")

    if len(vocab_lines) == 0:
        raise FileNotFoundError("No training data found across all folds to create the vocabulary.")

    # Save combined training logs to a temporary file
    combined_vocab_file = os.path.join(options["output_dir"], "combined_train_vocab.txt")
    with open(combined_vocab_file, 'w') as f:
        f.writelines(vocab_lines)

    # Create vocabulary from the combined training logs
    vocab = WordVocab(vocab_lines)
    print("vocab_size", len(vocab))
    vocab.save_vocab(options["vocab_path"])

    # Clean up the temporary combined vocab file
    os.remove(combined_vocab_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kfold", action="store_true", help="Enable K-Fold Cross Validation")
    parser.add_argument("--mode", choices=["train", "predict", "vocab"], required=True,
                        help="Mode of operation: train, predict, or vocab")

    # Prediction-specific arguments
    parser.add_argument("-m", "--mean", type=float, default=0, help="Gaussian mean for prediction")
    parser.add_argument("-s", "--std", type=float, default=1, help="Gaussian std for prediction")

    args = parser.parse_args()
    print("arguments", args)

    if args.mode == 'train' and args.kfold:
        # Train using K-Fold Cross-Validation
        train_k_fold(options)
    elif args.mode == 'train':
        # Normal training mode (not K-Fold)
        Trainer(options).train()
    elif args.mode == 'predict':
        # Prediction mode
        Predictor(options).predict()
    elif args.mode == 'vocab':
        # Vocabulary creation
        create_vocab(options)