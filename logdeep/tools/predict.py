#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import os
import sys
import time
from collections import Counter, defaultdict
sys.path.append('../../')

import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from logdeep.dataset.log import log_dataset
from logdeep.dataset.sample import session_window, sliding_window


def generate(fold_dir, name):
    """
    Load the specified data file from the given fold directory.

    Args:
        fold_dir (str): The path to the specific fold directory (e.g., 'fold_1').
        name (str): The name of the data file to load (e.g., 'eval').

    Returns:
        tuple: A tuple containing the loaded data iterator and its length.
    """
    file_path = os.path.join(fold_dir, name)  # Construct the file path using the fold directory
    print("Loading", file_path)
    with open(file_path, 'r') as f:
        data_iter = f.readlines()
    return data_iter, len(data_iter)


class Predicter():
    def __init__(self, model, options):
        self.output_dir = options['output_dir']
        self.device = options['device']
        self.model = model
        self.model_path = options['model_path']
        self.window_size = options['window_size']
        self.num_candidates = options['num_candidates']
        self.num_classes = options['num_classes']
        self.input_size = options['input_size']
        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.parameters = options['parameters']
        self.batch_size = options['batch_size']
        self.num_classes = options['num_classes']
        self.threshold = options["threshold"]
        self.gaussian_mean = options["gaussian_mean"]
        self.gaussian_std = options["gaussian_std"]
        self.save_dir = options['save_dir']
        self.is_logkey = options["is_logkey"]
        self.is_time = options["is_time"]
        self.vocab_path = options["vocab_path"]
        self.min_len = options["min_len"]
        self.test_ratio = options["test_ratio"]

    def detect_logkey_anomaly(self, output, label):
        num_anomaly = 0
        for i in range(len(label)):
            predicted = torch.argsort(output[i])[-self.num_candidates:].clone().detach().cpu()
            if label[i] not in predicted:
                num_anomaly += 1
        return num_anomaly

    def compute_anomaly(self, results, threshold=0):
        total_errors = 0
        for seq_res in results:
            if isinstance(threshold, float):
                threshold = seq_res["predicted_logkey"] * threshold

            error = (self.is_logkey and seq_res["logkey_anomaly"] > threshold) or \
                    (self.is_time and seq_res["params_anomaly"] > threshold)
            total_errors += int(error)

        return total_errors

    def find_best_threshold(self, test_normal_results, test_abnormal_results, threshold_range):
        test_abnormal_length = len(test_abnormal_results)
        test_normal_length = len(test_normal_results)
        res = [0, 0, 0, 0, 0, 0, 0, 0]  # th,tp, tn, fp, fn,  p, r, f1
        for th in threshold_range:
            FP = self.compute_anomaly(test_normal_results, th)
            TP = self.compute_anomaly(test_abnormal_results, th)
            if TP == 0:
                continue

            # Compute precision, recall and F1-measure
            TN = test_normal_length - FP
            FN = test_abnormal_length - TP
            P = 100 * TP / (TP + FP)
            R = 100 * TP / (TP + FN)
            F1 = 2 * P * R / (P + R)
            if F1 > res[-1]:
                res = [th, TP, TN, FP, FN, P, R, F1]
        return res

    def unsupervised_helper(self, model, data_iter, vocab, data_type, scale=None, min_len=0):
        test_results = []
        normal_errors = []

        num_test = len(data_iter)
        rand_index = torch.randperm(num_test)
        rand_index = rand_index[:int(num_test * self.test_ratio)]

        with torch.no_grad():
            for idx, line in tqdm(enumerate(data_iter)):
                if idx not in rand_index:
                    continue

                line = [ln.split(",") for ln in line.split()]

                if len(line) < min_len:
                    continue

                line = np.array(line)
                # if time duration exists in data
                if line.shape[1] == 2:
                    tim = line[:, 1].astype(float)
                    tim[0] = 0
                    logkey = line[:, 0]
                else:
                    logkey = line.squeeze()
                    # if time duration doesn't exist, then create a zero array for time
                    tim = np.zeros(logkey.shape)

                if scale is not None:
                    tim = np.array(tim).reshape(-1,1)
                    tim = scale.transform(tim).reshape(-1).tolist()

                logkeys, times = [logkey.tolist()], [tim.tolist()] # add next axis

                logs, labels = sliding_window((logkeys, times), vocab, window_size=self.window_size, is_train=False)
                dataset = log_dataset(logs=logs,
                                        labels=labels,
                                        seq=self.sequentials,
                                        quan=self.quantitatives,
                                        sem=self.semantics,
                                        param=self.parameters)
                data_loader = DataLoader(dataset,
                                               batch_size=min(len(dataset), 128),
                                               shuffle=True,
                                               pin_memory=True)
                # batch_size = len(dataset)
                num_logkey_anomaly = 0
                num_predicted_logkey = 0
                for _, (log, label) in enumerate(data_loader):
                    features = []
                    for value in log.values():
                        features.append(value.clone().detach().to(self.device))

                    output = model(features=features, device=self.device)

                    num_predicted_logkey += len(label)

                    num_logkey_anomaly += self.detect_logkey_anomaly(output, label)

                # result for line at idx
                result = {"logkey_anomaly":num_logkey_anomaly,
                          "predicted_logkey": num_predicted_logkey
                          }
                test_results.append(result)
                if idx < 10 or idx % 1000 == 0:
                    print(data_type, result)

            return test_results, normal_errors

    def predict_unsupervised(self):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))

        with open(self.vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        # Iterate through fold directories
        fold_dirs = [os.path.join(self.output_dir, fold) for fold in os.listdir(self.output_dir) if fold.startswith("fold_")]
        for fold_dir in fold_dirs:
            print(f"Processing fold: {fold_dir}...")

            # Load eval data from the fold directory
            test_loader, _ = generate(fold_dir, 'eval')

            scale = None
            if self.is_time:
                scale_path = os.path.join(fold_dir, "scale.pkl")
                with open(scale_path, "rb") as f:
                    scale = pickle.load(f)

            # Test the model
            start_time = time.time()
            test_results, normal_errors = self.unsupervised_helper(
                model, test_loader, vocab, 'eval', scale=scale, min_len=self.min_len)

            # Save the results in the fold directory
            results_path = os.path.join(fold_dir, "test_results.pkl")
            print("Saving test results to", results_path)
            with open(results_path, "wb") as f:
                pickle.dump(test_results, f)

            elapsed_time = time.time() - start_time
            print(f"Elapsed time for fold {fold_dir}: {elapsed_time:.2f} seconds")