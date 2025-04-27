from collections import Counter
import pickle


class Vocab(object):
    def __init__(self, logs, specials=['PAD', 'UNK']):
        self.pad_index = 0
        self.unk_index = 1

        self.stoi = {}
        self.itos = list(specials)

        event_count = Counter()
        for line in logs:
            for logkey in line.split():
                event_count[logkey] += 1

        for event, freq in event_count.items():
            self.itos.append(event)

        self.stoi = {e: i for i, e in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def save_vocab(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_vocab(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)


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
    vocab = Vocab(logs)
    vocab.save_vocab(output_file)
    print(f"Vocabulary saved to {output_file}. Total unique Event IDs: {len(vocab)}")


if __name__ == "__main__":
    import os

    # Directory containing k-fold splits
    fold_dir = "../output/bgl/"
    # Output file for the vocabulary
    output_file = "../output/bgl/vocab.pkl"

    # Generate the vocabulary
    generate_vocab(fold_dir, output_file)