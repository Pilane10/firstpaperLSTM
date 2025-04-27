# -*- coding: utf-8 -*-
import os
import json
import pickle
import torch
from logdeep.tools.predict import Predicter  # Import the Predicter class
from logdeep.models.lstm import Deeplog  # Import the Deeplog model
from logdeep.dataset.vocab import Vocab  # Import the Vocab class to resolve pickle error

def main():
    # Updated default directory structure to match "../output/bgl/"
    output_dir = os.path.join("..", "output", "bgl")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Config Parameters
    options = {
        "output_dir": output_dir,
        "device": device,
        "model_path": None,  # Will be updated for each fold
        "window_size": 20,
        "num_candidates": 9,
        "num_classes": 200,
        "input_size": 1,
        "sequentials": True,
        "quantitatives": False,
        "semantics": False,
        "parameters": False,
        "batch_size": 128,
        "threshold": None,
        "gaussian_mean": 0,
        "gaussian_std": 0,
        "save_dir": None,  # Will be updated for each fold
        "is_logkey": True,
        "is_time": False,
        "vocab_path": os.path.join(output_dir, "vocab.pkl"),
        "min_len": 10,
        "test_ratio": 1.0,  # Use the full test set
        "hidden_size": 64,
        "num_layers": 2,
        "embedding_dim": 50,
    }

    # Load vocabulary
    with open(options["vocab_path"], "rb") as f:
        vocab = pickle.load(f)  # Now resolved because Vocab is imported

    # Correctly scan for all fold directories in "../output/bgl/"
    fold_dirs = [os.path.join(output_dir, fold) for fold in os.listdir(output_dir) if fold.startswith("fold_")]
    fold_dirs.sort()  # Ensure consistent ordering

    all_fold_metrics = []

    for fold_dir in fold_dirs:
        print(f"Processing fold: {fold_dir}...")

        # Update model path and save directory for the current fold
        options["model_path"] = os.path.join(fold_dir, "deeplogbestloss.pth")
        options["save_dir"] = fold_dir

        # Instantiate the Deeplog model
        model = Deeplog(
            input_size=options["input_size"],
            hidden_size=options["hidden_size"],
            num_layers=options["num_layers"],
            vocab_size=options["num_classes"],
            embedding_dim=options["embedding_dim"],
        )

        # Instantiate the Predicter class
        predicter = Predicter(model, options)

        # Use Predicter to make predictions
        predicter.predict_unsupervised()

        # Load and aggregate results
        with open(os.path.join(fold_dir, "test_normal_results"), "rb") as f:
            test_normal_results = pickle.load(f)
        with open(os.path.join(fold_dir, "test_abnormal_results"), "rb") as f:
            test_abnormal_results = pickle.load(f)

        # Evaluate metrics
        threshold_range = range(10)  # Adjust as needed
        metrics = predicter.find_best_threshold(test_normal_results, test_abnormal_results, threshold_range)
        all_fold_metrics.append(metrics)

        print(f"Metrics for {fold_dir}:")
        print(json.dumps(metrics, indent=4))

    # Save aggregated results
    results_path = os.path.join(output_dir, "aggregated_results.json")
    with open(results_path, "w") as f:
        json.dump(all_fold_metrics, f, indent=4)

    print("Aggregated Results:")
    print(json.dumps(all_fold_metrics, indent=4))


if __name__ == "__main__":
    main()