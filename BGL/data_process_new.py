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
time_window = 60  # Size of the time window in seconds
n_folds = 5  # Number of k-folds
train_size = 5000
tuning_size_normal = 1000
tuning_size_anomaly = 1000
eval_size_normal = 5000
eval_size_anomaly = 1000

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

    # Debugging: Check the first few rows of the structured log
    print("Structured Log Head:\n", df.head())

    # Convert Label column
    df["Label"] = df["Label"].replace("-", 0).apply(lambda x: 1 if x != 0 else 0)

    # Debugging: Check unique values in the Label column
    print("Unique Labels after conversion:", df["Label"].unique())

    # Convert Time to datetime and create timestamp
    df["datetime"] = pd.to_datetime(df["Time"], format="%Y-%m-%d-%H.%M.%S.%f", errors="coerce")
    df["timestamp"] = df["datetime"].values.astype(np.int64) // 10 ** 9

    # Debugging: Check for missing or invalid timestamps
    print("Missing timestamps:", df["timestamp"].isnull().sum())

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

    # Debugging: Inspect a few grouped sequences
    print("Sample grouped sequences:\n", grouped_sequences[:5])

    return grouped_sequences


def generate_kfolds(grouped_sequences, output_dir, n_folds):
    """
    Generate k-fold splits and save to files.

    Args:
        grouped_sequences (list): Grouped sequences and labels.
        output_dir (str): Path to save the splits.
        n_folds (int): Number of k-folds.
    """
    normal_sequences = [seq for seq, label in grouped_sequences if label == 0]
    anomaly_sequences = [seq for seq, label in grouped_sequences if label == 1]

    print(f"Total normal sequences: {len(normal_sequences)}")
    print(f"Total anomaly sequences: {len(anomaly_sequences)}")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(normal_sequences), start=1):
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        train = [normal_sequences[idx] for idx in train_idx[:train_size]]
        tuning_normal = [normal_sequences[idx] for idx in train_idx[train_size:train_size + tuning_size_normal]]
        tuning_anomaly = anomaly_sequences[:tuning_size_anomaly]
        tuning = tuning_normal + tuning_anomaly

        eval_normal = [normal_sequences[idx] for idx in test_idx[:eval_size_normal]]
        eval_anomaly = anomaly_sequences[tuning_size_anomaly:tuning_size_anomaly + eval_size_anomaly]
        evaluation = eval_normal + eval_anomaly

        # Debugging: Check the sizes of the datasets
        print(f"Fold {fold_idx} - Train: {len(train)}, Tuning: {len(tuning)}, Evaluation: {len(evaluation)}")

        for dataset, name in zip([train, tuning, evaluation], ["train", "tuning", "eval"]):
            with open(os.path.join(fold_dir, name), "w") as f:
                for sequence in dataset:
                    f.write(" ".join(sequence) + "\n")
        print(f"Saved fold {fold_idx} to {fold_dir}")


if __name__ == "__main__":
    # Step 1: Parse raw logs
    parse_log(data_dir, output_dir, log_file)

    # Step 2: Preprocess logs
    structured_log_path = os.path.join(output_dir, f"{log_file}_structured.csv")
    df = preprocess_logs(structured_log_path)

    # Debugging: Check the overall structure of the DataFrame
    print("Preprocessed DataFrame Info:")
    print(df.info())

    # Step 3: Group logs into sequences
    grouped_sequences = group_logs(df, time_window)

    # Step 4: Generate k-fold splits
    generate_kfolds(grouped_sequences, output_dir, n_folds)

    print("Data processing and k-fold generation completed.")