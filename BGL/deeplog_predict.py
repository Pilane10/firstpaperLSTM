# -*- coding: utf-8 -*-
import os
import json
import pickle
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


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


def load_model(checkpoint_path, model, device):
    """
    Load the model checkpoint.
    """
    print(f"Loading model checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    print("Model loaded successfully.")
    return model


def predict_and_evaluate_fold(model, test_loader, device):
    """
    Perform predictions and evaluate metrics for a single fold.
    """
    model.eval()
    all_probs = []
    all_labels = []
    predicted_labels = []

    # Predict on Test Set
    print("Predicting on test data...")
    with torch.no_grad():
        for logs, labels in tqdm(test_loader):
            features = []
            for value in logs.values():
                features.append(value.clone().detach().to(device))

            output = model(features=features, device=device)
            output = output.squeeze()

            probs = torch.softmax(output, dim=-1)  # Normalize scores to probabilities
            scores = torch.argmax(probs, dim=-1)  # Predicted class labels

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(scores.cpu().numpy())

    # Ensure y_score covers all classes in y_true
    unique_classes = np.unique(all_labels)
    num_classes = len(unique_classes)
    y_score = np.zeros((len(all_probs), num_classes))  # Create a zeroed matrix for all classes

    # Populate y_score with probabilities for present classes
    for i, prob in enumerate(all_probs):
        for class_idx, class_label in enumerate(unique_classes):
            if class_label < len(prob):  # Check if the class exists in the model's output
                y_score[i, class_idx] = prob[class_label]

    # Compute Metrics
    metrics = {
        "precision": precision_score(
            all_labels, predicted_labels, average="macro", zero_division=0
        ),
        "recall": recall_score(
            all_labels, predicted_labels, average="macro", zero_division=0
        ),
        "f1_score": f1_score(
            all_labels, predicted_labels, average="macro", zero_division=0
        ),
        "confusion_matrix": confusion_matrix(all_labels, predicted_labels).tolist()
    }

    return metrics, all_probs, all_labels


def aggregate_results(fold_metrics):
    """
    Aggregate results across all folds.
    """
    aggregated_metrics = defaultdict(list)

    # Collect metrics across folds
    for metrics in fold_metrics:
        for key, value in metrics.items():
            if key != "confusion_matrix":
                aggregated_metrics[key].append(value)

    # Aggregate metrics
    aggregated_results = {
        key: {
            "mean": np.mean(values),
            "std": np.std(values)
        } for key, values in aggregated_metrics.items()
    }

    return aggregated_results


def main():
    # Updated default directory structure to match "../output/bgl/"
    output_dir = os.path.join("..", "output", "bgl")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128

    from logdeep.models.lstm import Deeplog
    from logdeep.dataset.log import log_dataset
    from logdeep.dataset.sample import sliding_window

    model = Deeplog(input_size=1, hidden_size=64, num_layers=2, vocab_size=200, embedding_dim=50)
    model = model.to(device)

    # Load vocabulary
    vocab_path = os.path.join(output_dir, "vocab.pkl")
    vocab = Vocab.load_vocab(vocab_path)

    # Correctly scan for all fold directories in "../output/bgl/"
    fold_dirs = [os.path.join(output_dir, fold) for fold in os.listdir(output_dir) if fold.startswith("fold_")]
    fold_dirs.sort()  # Ensure consistent ordering

    all_fold_metrics = []

    for fold_dir in fold_dirs:
        print(f"Processing fold: {fold_dir}...")

        # Load model for the current fold
        model_path = os.path.join(fold_dir, "deeplogbestloss.pth")  # Corrected model path
        model = load_model(model_path, model, device)

        # Load test data
        test_data_path = os.path.join(fold_dir, "eval")
        with open(test_data_path, "r") as f:
            test_sequences = [line.strip().split() for line in f.readlines()]

        # Prepare test dataset
        logkeys, times = test_sequences, [[0] * len(seq) for seq in test_sequences]  # Generate dummy times if not available
        test_logs, test_labels = sliding_window((logkeys, times), vocab=vocab, window_size=20)  # Pass as tuple
        test_dataset = log_dataset(logs=test_logs, labels=test_labels, seq=True, quan=False, sem=False, param=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Predict and evaluate
        metrics, _, _ = predict_and_evaluate_fold(model, test_loader, device)
        all_fold_metrics.append(metrics)

        print(f"Metrics for {fold_dir}:")
        print(json.dumps(metrics, indent=4))

    # Aggregate results across folds
    aggregated_results = aggregate_results(all_fold_metrics)

    # Save aggregated results
    results_path = os.path.join(output_dir, "aggregated_results.json")
    with open(results_path, "w") as f:
        json.dump(aggregated_results, f, indent=4)

    print("Aggregated Results:")
    print(json.dumps(aggregated_results, indent=4))


if __name__ == "__main__":
    main()