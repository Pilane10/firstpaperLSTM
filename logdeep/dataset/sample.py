import os
import json
from collections import Counter
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

def read_json(filename):
    with open(filename, 'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict


def trp(l, n):
    """ Truncate or pad a list """
    r = l[:n]
    if len(r) < n:
        r.extend(list([0]) * (n - len(r)))
    return r


def down_sample(logs, labels, sample_ratio):
    print('sampling...')
    total_num = len(labels)
    all_index = list(range(total_num))
    sample_logs = {}
    for key in logs.keys():
        sample_logs[key] = []
    sample_labels = []
    sample_num = int(total_num * sample_ratio)

    for i in tqdm(range(sample_num)):
        random_index = int(np.random.uniform(0, len(all_index)))
        for key in logs.keys():
            sample_logs[key].append(logs[key][random_index])
        sample_labels.append(labels[random_index])
        del all_index[random_index]
    return sample_logs, sample_labels


# https://stackoverflow.com/questions/15357422/python-determine-if-a-string-should-be-converted-into-int-or-float
def isfloat(x):
    try:
        a = float(x)
    except ValueError:
        return False
    else:
        return True

def isint(x):
    try:
        a = float(x)
        b = int(a)
    except ValueError:
        return False
    else:
        return a == b


def split_features(data_path, train_ratio=1, scale=None, scale_path=None, min_len=0):
    """
    Splits features from the dataset file.

    Args:
        data_path (str): Path to the dataset file.
        train_ratio (float): Ratio of data to use for training.
        scale (object): Scaler for time normalization.
        scale_path (str): Path to save the scaler.
        min_len (int): Minimum length of a sequence to include.

    Returns:
        tuple: logkeys and times
    """
    # Ensure the path is correct
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"No such file or directory: '{data_path}'")

    with open(data_path, 'r') as f:
        data = f.readlines()

    sample_size = int(len(data) * train_ratio)
    data = data[:sample_size]
    logkeys = []
    times = []

    for line in data:
        line = [ln.split(",") for ln in line.split()]

        if len(line) < min_len:
            continue

        line = np.array(line)
        # If time duration exists in data
        if line.shape[1] == 2:
            tim = line[:, 1].astype(float)
            tim[0] = 0
            logkey = line[:, 0]
        else:
            logkey = line.squeeze()
            # If time duration doesn't exist, then create a zero array for time
            tim = np.zeros(logkey.shape)

        logkeys.append(logkey.tolist())
        times.append(tim.tolist())

    if scale is not None:
        total_times = np.concatenate(times, axis=0).reshape(-1, 1)
        scale.fit(total_times)

        for i, tn in enumerate(times):
            tn = np.array(tn).reshape(-1, 1)
            times[i] = scale.transform(tn).reshape(-1).tolist()

        with open(scale_path, 'wb') as f:
            pickle.dump(scale, f)
        print(f"Save scale at {scale_path}\n")

    return logkeys, times


def sliding_window(data_iter, vocab, window_size, is_train=True):
    """
    Creates sliding window datasets.

    Args:
        data_iter (iterable): Iterable containing log data and parameters.
        vocab (object): Vocabulary object for encoding.
        window_size (int): Size of the sliding window.
        is_train (bool): Whether the operation is for training.

    Returns:
        tuple: Resulting logs and labels.
    """
    result_logs = {
        'Sequentials': [],
        'Quantitatives': [],
        'Semantics': [],
        'Parameters': []
    }
    labels = []

    num_sessions = 0
    num_classes = len(vocab)

    for line, params in zip(*data_iter):
        if num_sessions % 1000 == 0:
            print(f"Processed {num_sessions} lines", end='\r')
        num_sessions += 1

        line = [vocab.stoi.get(ln, vocab.unk_index) for ln in line]

        session_len = max(len(line), window_size) + 1  # Predict the next one
        padding_size = session_len - len(line)
        params = params + [0] * padding_size
        line = line + [vocab.pad_index] * padding_size

        for i in range(session_len - window_size):
            Parameter_pattern = params[i:i + window_size]
            Sequential_pattern = line[i:i + window_size]
            Semantic_pattern = []

            Quantitative_pattern = [0] * num_classes
            log_counter = Counter(Sequential_pattern)

            for key in log_counter:
                Quantitative_pattern[key] = log_counter[key]

            Sequential_pattern = np.array(Sequential_pattern)
            Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]

            result_logs['Sequentials'].append(Sequential_pattern)
            result_logs['Quantitatives'].append(Quantitative_pattern)
            result_logs['Semantics'].append(Semantic_pattern)
            result_logs["Parameters"].append(Parameter_pattern)
            labels.append(line[i + window_size])

    if is_train:
        print(f'Number of sessions: {num_sessions}')
        print(f'Number of seqs: {len(result_logs["Sequentials"])}')

    return result_logs, labels


def session_window(data_dir, datatype, sample_ratio=1):
    """
    Processes sessions using session-based windows.

    Args:
        data_dir (str): Path to the data directory.
        datatype (str): Type of data ('train', 'val', 'test').
        sample_ratio (float): Down-sampling ratio.

    Returns:
        tuple: Resulting logs and labels.
    """
    event2semantic_vec = read_json(os.path.join(data_dir, 'hdfs/event2semantic_vec.json'))
    result_logs = {
        'Sequentials': [],
        'Quantitatives': [],
        'Semantics': []
    }
    labels = []

    data_file = None
    if datatype == 'train':
        data_file = os.path.join(data_dir, 'hdfs/robust_log_train.csv')
    elif datatype == 'val':
        data_file = os.path.join(data_dir, 'hdfs/robust_log_valid.csv')
    elif datatype == 'test':
        data_file = os.path.join(data_dir, 'hdfs/robust_log_test.csv')

    if not data_file or not os.path.isfile(data_file):
        raise FileNotFoundError(f"No such file: '{data_file}'")

    train_df = pd.read_csv(data_file)
    for i in tqdm(range(len(train_df))):
        ori_seq = [int(eventid) for eventid in train_df["Sequence"][i].split(' ')]
        Sequential_pattern = trp(ori_seq, 50)
        Semantic_pattern = []
        for event in Sequential_pattern:
            if event == 0:
                Semantic_pattern.append([-1] * 300)
            else:
                Semantic_pattern.append(event2semantic_vec[str(event - 1)])
        Quantitative_pattern = [0] * 29
        log_counter = Counter(Sequential_pattern)

        for key in log_counter:
            Quantitative_pattern[key] = log_counter[key]

        Sequential_pattern = np.array(Sequential_pattern)
        Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]
        result_logs['Sequentials'].append(Sequential_pattern)
        result_logs['Quantitatives'].append(Quantitative_pattern)
        result_logs['Semantics'].append(Semantic_pattern)
        labels.append(int(train_df["label"][i]))

    if sample_ratio != 1:
        result_logs, labels = down_sample(result_logs, labels, sample_ratio)

    print(f'Number of sessions({data_file}): {len(result_logs["Semantics"])}')
    return result_logs, labels