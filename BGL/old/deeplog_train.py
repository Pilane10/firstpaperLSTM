import os
import pickle
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from logdeep.models.lstm import Deeplog  # Import the Deeplog model

# Configuration options
options = {
    # Dataset
    "window_size": 20,  # Sliding window size
    "min_len": 10,  # Minimum sequence length

    # Model
    "input_size": 1,
    "hidden_size": 64,
    "num_layers": 2,
    "embedding_dim": 50,
    "vocab_size": 200,  # This will be updated dynamically based on the vocabulary

    # Training
    "batch_size": 8, #128
    "learning_rate": 0.01,
    "max_epoch": 10,
    "optimizer": "adam",

    # Directories
    "fold_dir": "../output/bgl/",
    "vocab_file": "../output/bgl/vocab.pkl",
    "results_dir": "../output/bgl/deeplog/results",
}

# Dataset class with validation
class DeepLogDataset(Dataset):
    def __init__(self, file_path, vocab, min_len=10):
        """
        Initialize the dataset with validation.

        Args:
            file_path (str): Path to the training file.
            vocab (dict): Vocabulary mapping Event IDs to numerical IDs.
            min_len (int): Minimum sequence length.
        """
        self.data = []
        self.vocab_size = len(vocab)
        with open(file_path, "r") as f:
            for line in f:
                event_ids = line.strip().split()
                # Map Event IDs to indices and validate
                mapped_ids = [vocab[event_id] for event_id in event_ids if event_id in vocab]

                # Check for out-of-range indices
                if any(idx >= self.vocab_size for idx in mapped_ids):
                    print(f"Skipping sequence with out-of-range indices: {mapped_ids}")
                    continue

                # Skip sequences shorter than the minimum length
                if len(mapped_ids) >= min_len:
                    self.data.append(mapped_ids)
                else:
                    print(f"Skipping short sequence: {event_ids}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        return torch.tensor(sequence[:-1], dtype=torch.long), torch.tensor(sequence[-1], dtype=torch.long)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_deeplog(options):
    """
    Train DeepLog using configurations from the options dictionary.

    Args:
        options (dict): Configuration settings for the model, dataset, and training.
    """
    # Load vocabulary
    with open(options["vocab_file"], "rb") as f:
        vocab = pickle.load(f)
    options["vocab_size"] = len(vocab)
    print(f"Vocabulary size: {options['vocab_size']}")  # Log vocabulary size

    # Create results directory if it doesn't exist
    os.makedirs(options["results_dir"], exist_ok=True)

    # Iterate over fold directories
    for fold_name in os.listdir(options["fold_dir"]):
        if not fold_name.startswith("fold_"):
            continue

        fold_path = os.path.join(options["fold_dir"], fold_name)
        if not os.path.isdir(fold_path):
            continue

        print(f"Training on {fold_name}...")
        fold_results_dir = os.path.join(options["results_dir"], fold_name)
        os.makedirs(fold_results_dir, exist_ok=True)

        # Load train dataset
        train_file = os.path.join(fold_path, "train")
        if not os.path.isfile(train_file):
            print(f"Train file not found in {fold_name}, skipping...")
            continue

        train_dataset = DeepLogDataset(train_file, vocab, min_len=options["min_len"])
        train_loader = DataLoader(train_dataset, batch_size=options["batch_size"], shuffle=True)

        # Initialize model, loss function, and optimizer
        model = Deeplog(
            input_size=options["input_size"],
            hidden_size=options["hidden_size"],
            num_layers=options["num_layers"],
            vocab_size=options["vocab_size"],
            embedding_dim=options["embedding_dim"]
        )
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"])

        # Training loop
        model.train()
        for epoch in range(options["max_epoch"]):
            epoch_loss = 0
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{options['max_epoch']}") as pbar:
                for batch in train_loader:
                    if len(batch) == 0:  # Handle empty batches
                        print("Skipping empty batch")
                        continue

                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)

                    # Debug: Print input indices
                    print(f"Input indices (batch): {inputs}")

                    optimizer.zero_grad()
                    outputs = model([inputs], device)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    pbar.update(1)

            print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}")

            # Save the model after each epoch
            torch.save(model.state_dict(), os.path.join(fold_results_dir, f"deeplog_model_epoch_{epoch + 1}.pth"))

        # Save the final model and training logs
        torch.save(model.state_dict(), os.path.join(fold_results_dir, "deeplog_model.pth"))
        with open(os.path.join(fold_results_dir, "training_log.txt"), "w") as log_file:
            log_file.write(f"Training completed for {fold_name}\n")
            log_file.write(f"Final Epoch Loss: {epoch_loss:.4f}\n")

    print("Training completed for all folds.")


# Add debugging logs in the model
class DebugDeeplog(Deeplog):
    def forward(self, x, device):
        embed0 = self.embedding(x[0].to(device))
        print(f"Embedding size: {embed0.size()}")  # Log embedding size
        h0 = torch.zeros(self.num_layers, embed0.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, embed0.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(embed0, (h0, c0))
        print(f"LSTM output size: {out.size()}")  # Log LSTM output size

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        print(f"FC output size: {out.size()}")  # Log fully connected output size

        return out


if __name__ == "__main__":
    train_deeplog(options)