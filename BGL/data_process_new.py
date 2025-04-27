import os
import gc
import pandas as pd
import numpy as np
from logparser import Drain
from sklearn.model_selection import KFold
from tqdm import tqdm

# Configuration
data_dir = os.path.expanduser("~/.dataset/bgl")
output_dir = "../output/bgl/"
log_file = "BGL.log"
time_window = 60  # Time window size in seconds
n_folds = 5  # Number of k-folds
total_normal_logs = 25000  # Total normal logs
total_anomaly_logs = 1800  # Total anomaly logs
train_split_size = 20000  # Training split size for normal logs
train_size_per_fold = 5000  # Training size per fold for normal logs
tuning_size_normal = 1000  # Tuning size for normal logs (from training split)
tuning_size_anomaly = 1000  # Tuning size for anomaly logs
eval_size_normal = 5000  # Evaluation size for normal logs
eval_size_anomaly = 1000  # Evaluation size for anomaly logs

# Constants for Label
PAD = 0
UNK = 1
START = 2

def parse_log(input_dir, output_dir, log_file):
    """
    Parse the raw log file using the Drain parser and structure it into a CSV file.

    Args:
        input_dir (str): Path to the raw log directory.
        output_dir (str): Path to the output directory for structured logs.
        log_file (str): Name of the raw log file.
    """
    log_format = '<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>'
    regex = [
        r'(0x)[0-9a-fA-F]+',  # hexadecimal
        r'\d+\.\d+\.\d+\.\d+',  # IP addresses
        r'\d+'  # numbers
    ]
    keep_para = False

    # Drain parser configuration
    st = 0.3  # Similarity threshold
    depth = 3  # Depth of all leaf nodes
    parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex, keep_para=keep_para)
    parser.parse(log_file)

    print(f"Logs parsed and structured log saved at {output_dir}")


def preprocess_logs(structured_log_path):
    """
    Preprocess the structured log file by cleaning and converting columns.

    Args:
        structured_log_path (str): Path to the structured log file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df = pd.read_csv(structured_log_path)

    # Convert Label column
    df["Label"] = df["Label"].replace("-", 0).apply(lambda x: 1 if x != 0 else 0)

    # Convert Time to datetime and create timestamp
    df["datetime"] = pd.to_datetime(df["Time"], format="%Y-%m-%d-%H.%M.%S.%f", errors="coerce")
    df["timestamp"] = df["datetime"].values.astype(np.int64) // 10 ** 9

    return df


def group_logs(df, time_window):
    """
    Group logs into sequences based on time windows.

    Args:
        df (pd.DataFrame): Preprocessed log DataFrame.
        time_window (int): Time window size in seconds.

    Returns:
        list: Grouped sequences and their labels.
    """
    df["TimeWindow"] = (df["timestamp"] - df["timestamp"].min()) // time_window
    grouped_sequences = []

    for _, group in tqdm(df.groupby("TimeWindow"), desc="Grouping Logs"):
        sequence = group["EventId"].tolist()
        label = 1 if any(group["Label"] == 1) else 0
        grouped_sequences.append((sequence, label))

    return grouped_sequences


def generate_kfolds(grouped_sequences, output_dir, n_folds):
    """
    Generate k-fold splits and save to files.

    Args:
        grouped_sequences (list): Grouped sequences and labels.
        output_dir (str): Path to save the splits.
        n_folds (int): Number of k-folds.
    """
    normal_sequences = [seq for seq, label in grouped_sequences if label == 0][:total_normal_logs]
    anomaly_sequences = [seq for seq, label in grouped_sequences if label == 1][:total_anomaly_logs]

    print(f"Total normal sequences: {len(normal_sequences)}")
    print(f"Total anomaly sequences: {len(anomaly_sequences)}")

    # Split normal logs into train and evaluation sets
    train_normal_logs = normal_sequences[:train_split_size]  # 20,000 normal logs for training
    eval_normal_logs = normal_sequences[train_split_size:]  # 5,000 normal logs for evaluation

    # K-Fold Cross-Validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold_idx, (train_idx, _) in enumerate(kf.split(train_normal_logs), start=1):
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        # Training data (5,000 normal logs per fold)
        train = [train_normal_logs[idx] for idx in train_idx[:train_size_per_fold]]

        # Tuning data (1,000 normal + 1,000 anomaly logs)
        tuning_normal = [train_normal_logs[idx] for idx in train_idx[train_size_per_fold:train_size_per_fold + tuning_size_normal]]
        tuning_anomaly = anomaly_sequences[:tuning_size_anomaly]
        tuning = tuning_normal + tuning_anomaly

        # Evaluation data (5,000 normal + 1,000 anomaly logs)
        evaluation = eval_normal_logs + anomaly_sequences[tuning_size_anomaly:tuning_size_anomaly + eval_size_anomaly]

        # Save the datasets and print their sizes
        for dataset, name in zip([train, tuning, evaluation], ["train", "tuning", "eval"]):
            dataset_path = os.path.join(fold_dir, name)
            with open(dataset_path, "w") as f:
                for sequence in dataset:
                    f.write(" ".join(sequence) + "\n")
            print(f"Fold {fold_idx} - {name.capitalize()} Size: {len(dataset)} logs")

        print(f"Saved fold {fold_idx} to {fold_dir}")


if __name__ == "__main__":
    # Step 1: Parse raw logs
    parse_log(data_dir, output_dir, log_file)

    # Step 2: Preprocess logs
    structured_log_path = os.path.join(output_dir, f"{log_file}_structured.csv")
    df = preprocess_logs(structured_log_path)

    # Step 3: Group logs into sequences
    grouped_sequences = group_logs(df, time_window)

    # Step 4: Generate k-fold splits
    generate_kfolds(grouped_sequences, output_dir, n_folds)

    print("Data processing and k-fold generation completed.")