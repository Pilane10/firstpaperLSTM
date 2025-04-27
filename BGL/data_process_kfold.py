import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

output_dir = "../output/bgl/deeplog/"
log_file = "../output/bgl/BGL.log_structured.csv"


def time_window_grouping(logs, time_window):
    """
    Group log events into sequences based on specified time windows.

    Args:
        logs (DataFrame): Structured log data with columns 'Time', 'EventId', and 'Label'.
        time_window (int): Size of the time window in seconds.

    Returns:
        list of tuples: A list of (sequence, label) where sequence is a list of Event IDs and label is the sequence label.
    """
    # Convert time column to datetime
    logs["Time"] = pd.to_datetime(logs["Time"], format="%Y-%m-%d-%H.%M.%S.%f", errors="coerce")
    if logs["Time"].isnull().any():
        raise ValueError("Some timestamps could not be parsed. Check the 'Time' column for invalid formats.")

    # Create time windows
    logs["TimeWindow"] = (logs["Time"] - logs["Time"].min()).dt.total_seconds() // time_window

    # Group logs by time window
    grouped_sequences = []
    for _, group in tqdm(logs.groupby("TimeWindow"), desc="Time Window Grouping"):
        sequence = group["EventId"].tolist()
        label = 1 if any(group["Label"] == 1) else 0
        grouped_sequences.append((sequence, label))

    # Debugging: Check for NoneType values in grouped sequences
    for sequence, label in grouped_sequences:
        if any(event_id is None for event_id in sequence):
            print(f"NoneType found in grouped sequence: {sequence}")

    return grouped_sequences


def generate_kfold_splits(grouped_sequences, k=5):
    """
    Generate k-fold splits with specific dataset allocations:
    - Training: 5,000 normal logs.
    - Hyperparameter Tuning: 1,000 normal logs + 1,000 anomaly logs.
    - Evaluation: 5,000 normal logs + 1,000 anomaly logs.

    Args:
        grouped_sequences (list of tuples): List of (sequence, label).
        k (int): Number of folds.

    Returns:
        list: K-fold splits containing train, tuning, and evaluation sets.
    """
    # Separate normal and anomaly sequences
    normal_sequences = [seq for seq, label in grouped_sequences if label == 0]
    anomaly_sequences = [seq for seq, label in grouped_sequences if label == 1]

    # Debugging: Print available sequences
    print(f"Total normal sequences: {len(normal_sequences)}")
    print(f"Total anomaly sequences: {len(anomaly_sequences)}")

    # Relaxed requirements
    required_normal = 24700  # Relaxed from 25,000
    required_anomaly = 1700  # Relaxed from 2,000

    # Ensure there are enough sequences for the splits
    if len(normal_sequences) < required_normal or len(anomaly_sequences) < required_anomaly:
        raise ValueError(
            f"Not enough sequences for the described k-fold process. "
            f"Normal logs available: {len(normal_sequences)}, required: {required_normal}. "
            f"Anomaly logs available: {len(anomaly_sequences)}, required: {required_anomaly}."
        )

    # Sample the required number of sequences
    normal_sequences = pd.DataFrame(normal_sequences).sample(n=required_normal, random_state=42).values.tolist()
    anomaly_sequences = pd.DataFrame(anomaly_sequences).sample(n=required_anomaly, random_state=42).values.tolist()

    # Allocation criteria
    train_size = 5000
    tuning_size_normal = 1000
    tuning_size_anomaly = 1000
    eval_size_normal = 5000
    eval_size_anomaly = 1000

    # Prepare k-fold splits
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    folds = []

    print("Generating k-fold splits...")
    for fold, (train_idx, test_idx) in tqdm(enumerate(kf.split(normal_sequences), start=1), total=k, desc="K-Fold Splits"):
        # Training: 5,000 normal logs from the training split
        train_data = [normal_sequences[idx] for idx in train_idx[:train_size]]

        # Hyperparameter Tuning: 1,000 normal logs + 1,000 anomaly logs
        tuning_normal = [normal_sequences[idx] for idx in train_idx[train_size:train_size + tuning_size_normal]]
        tuning_anomaly = anomaly_sequences[:tuning_size_anomaly]
        tuning_data = tuning_normal + tuning_anomaly

        # Evaluation: 5,000 normal logs + 1,000 anomaly logs
        eval_normal = [normal_sequences[idx] for idx in test_idx[:eval_size_normal]]
        eval_anomaly = [anomaly_sequences[idx] for idx in range(eval_size_anomaly)]
        eval_data = eval_normal + eval_anomaly

        # Replace NoneType values in sequences with "UNK"
        train_data = [[event_id if event_id is not None else "UNK" for event_id in sequence] for sequence in train_data]
        tuning_data = [[event_id if event_id is not None else "UNK" for event_id in sequence] for sequence in tuning_data]
        eval_data = [[event_id if event_id is not None else "UNK" for event_id in sequence] for sequence in eval_data]

        folds.append((train_data, tuning_data, eval_data))

    return folds


if __name__ == "__main__":
    ##################
    # Transformation #
    ##################
    # Load the structured log file generated by the original data_process.py
    df = pd.read_csv(log_file)

    # Inspect and handle non-numeric values in the Label column
    print("Unique values in Label column before processing:", df["Label"].unique())
    df["Label"] = df["Label"].replace("-", 0)  # Replace "-" with 0 for normal logs
    df["Label"] = df["Label"].apply(lambda x: 1 if x != 0 else 0)  # Convert non-zero values to 1
    print("Processed values in Label column:", df["Label"].unique())

    # Handle missing EventId values
    print("Checking for missing values in EventId:")
    print(df["EventId"].isnull().sum())
    df["EventId"] = df["EventId"].fillna("UNK")  # Replace NaN with "UNK"

    # Debugging: Count normal and anomaly logs
    normal_count = len(df[df["Label"] == 0])
    anomaly_count = len(df[df["Label"] == 1])

    print(f"Total normal logs: {normal_count}")
    print(f"Total anomaly logs: {anomaly_count}")

    # Check if conditions are met
    if normal_count < 24700 or anomaly_count < 1700:
        print("Not enough normal or anomaly logs to meet the requirements.")
        exit(1)

    # Group logs into sequences based on time windows
    time_window = 300  # Time window size in seconds (e.g., 10, 30, 60)
    grouped_sequences = time_window_grouping(df, time_window)

    # Generate k-fold splits
    k = 5  # Number of folds
    folds = generate_kfold_splits(grouped_sequences, k)

    # Save the splits
    print("Saving k-fold splits...")
    for fold_idx, (train, tuning, eval) in tqdm(enumerate(folds, start=1), total=k, desc="Saving Splits"):  # Start fold numbering at 1
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        # Save files in the required format
        for dataset, name in zip([train, tuning, eval], ["train", "tuning", "eval"]):
            with open(os.path.join(fold_dir, name), "w") as f:
                for sequence in dataset:
                    f.write(" ".join(sequence) + "\n")

        print(f"Saved Fold {fold_idx} in {fold_dir}")

    print("Time-based grouping and k-fold split generation completed.")