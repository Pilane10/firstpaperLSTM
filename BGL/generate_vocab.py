import os
import pickle
from logdeep.dataset.vocab import Vocab  # Use the Vocab class from logdeep

def generate_vocab(fold_dir, output_file):
    """
    Generate a vocabulary mapping Event IDs to numerical IDs.

    Args:
        fold_dir (str): Path to the directory containing k-fold splits (fold_1, fold_2, etc.).
        output_file (str): Path to save the generated vocab.pkl file.
    """
    logs = []
    # Iterate over all fold directories
    for fold_name in os.listdir(fold_dir):
        fold_path = os.path.join(fold_dir, fold_name)
        if not os.path.isdir(fold_path):
            continue

        # Process the train file in each fold
        train_file = os.path.join(fold_path, "train")
        if not os.path.isfile(train_file):
            continue

        # Read the train file and collect logs
        with open(train_file, "r") as f:
            logs.extend(f.readlines())

    # Create the vocabulary
    vocab = Vocab(logs)  # Use the imported Vocab class
    vocab.save_vocab(output_file)
    print(f"Vocabulary saved to {output_file}. Total unique Event IDs: {len(vocab)}")


if __name__ == "__main__":
    # Directory containing k-fold splits
    fold_dir = "../output/bgl/"
    # Output file for the vocabulary
    output_file = "../output/bgl/vocab.pkl"

    # Generate the vocabulary
    generate_vocab(fold_dir, output_file)