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


def generate(output_dir, name):
    print("Loading", os.path.join(output_dir, name))
    with open(os.path.join(output_dir, name), 'r') as f:
        data_iter = f.readlines()
    return data_iter, len(data_iter)


class Predicter():
    def __init__(self, model, options, config):
        self.output_dir = options['output_dir']
        self.device = options['device']
        self.model = model
        self.config = config
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
        self.threshold = options["threshold"]
        self.gaussian_mean = options["gaussian_mean"]
        self.gaussian_std = options["gaussian_std"]
        self.save_dir = options['save_dir']
        self.is_logkey = options["is_logkey"]
        self.is_time = options["is_time"]
        self.vocab_path = options["vocab_path"]
        self.min_len = options["min_len"]
        self.test_ratio = options["test_ratio"]
        self.pos_weight = options.get('pos_weight', 5.0)

    def detect_logkey_anomaly(self, output, label):
        """Detect anomalies based on next-token prediction"""
        next_pred, anomaly_score = output
        num_anomaly = 0
        for i in range(len(label)):
            predicted = torch.argsort(next_pred[i])[-self.num_candidates:].clone().detach().cpu()
            if label[i] not in predicted:
                num_anomaly += 1
        return num_anomaly

    def compute_anomaly(self, results, threshold=0):
        """Compute anomalies using both prediction and anomaly score"""
        total_errors = 0
        for seq_res in results:
            # Use anomaly score if available, otherwise fall back to logkey prediction
            if 'anomaly_score' in seq_res:
                error = seq_res['anomaly_score'] > threshold
            else:
                if isinstance(threshold, float):
                    threshold = seq_res["predicted_logkey"] * threshold
                error = (self.is_logkey and seq_res["logkey_anomaly"] > threshold)
            total_errors += int(error)
        return total_errors

    def find_best_threshold(self, test_normal_results, test_abnormal_results, threshold_range):
        test_abnormal_length = len(test_abnormal_results)
        test_normal_length = len(test_normal_results)
        best_metrics = [0, 0, 0, 0, 0, 0, 0, 0]  # th, tp, tn, fp, fn, p, r, f1

        for th in threshold_range:
            FP = self.compute_anomaly(test_normal_results, th)
            TP = self.compute_anomaly(test_abnormal_results, th)

            if TP == 0:
                continue

            # Compute metrics
            TN = test_normal_length - FP
            FN = test_abnormal_length - TP
            P = 100 * TP / (TP + FP)
            R = 100 * TP / (TP + FN)
            F1 = 2 * P * R / (P + R)

            if F1 > best_metrics[-1]:
                best_metrics = [th, TP, TN, FP, FN, P, R, F1]
        return best_metrics

    def unsupervised_helper(self, model, data_iter, vocab, data_type, scale=None, min_len=0):
        test_results = []
        normal_errors = []
        num_test = len(data_iter)
        rand_index = torch.randperm(num_test)
        rand_index = rand_index[:int(num_test * self.test_ratio)]

        with torch.no_grad():
            for idx, line in tqdm(enumerate(data_iter), desc=f"Processing {data_type}"):
                if idx not in rand_index:
                    continue

                line = [ln.split(",") for ln in line.split()]
                if len(line) < min_len:
                    continue

                line = np.array(line)
                if line.shape[1] == 2:  # If time duration exists
                    tim = line[:, 1].astype(float)
                    tim[0] = 0
                    logkey = line[:, 0]
                else:
                    logkey = line.squeeze()
                    tim = np.zeros(logkey.shape)

                if scale is not None:
                    tim = np.array(tim).reshape(-1, 1)
                    tim = scale.transform(tim).reshape(-1).tolist()

                logkeys, times = [logkey.tolist()], [tim.tolist()]
                logs, labels = sliding_window((logkeys, times), vocab,
                                              window_size=self.window_size,
                                              is_train=False)

                dataset = log_dataset(
                    logs=logs,
                    labels=labels,
                    seq=self.sequentials,
                    quan=self.quantitatives,
                    sem=self.semantics,
                    param=self.parameters
                )

                data_loader = DataLoader(
                    dataset,
                    batch_size=min(len(dataset), 128),
                    shuffle=False,
                    pin_memory=True
                )

                batch_logkey_anomaly = 0
                batch_predicted_logkey = 0
                batch_anomaly_scores = []

                for _, (log, label) in enumerate(data_loader):
                    features = []
                    for value in log.values():
                        features.append(value.clone().detach().to(self.device))

                    next_pred, anomaly_score = model(*features)

                    batch_predicted_logkey += len(label)
                    batch_logkey_anomaly += self.detect_logkey_anomaly((next_pred, anomaly_score), label)

                    # Handle anomaly score extraction properly
                    score = anomaly_score.squeeze()
                    if score.dim() == 0:  # Single element case
                        batch_anomaly_scores.append(score.item())
                    else:
                        batch_anomaly_scores.extend(score.cpu().numpy())

                # Store results for this sequence
                avg_anomaly_score = np.mean(batch_anomaly_scores) if batch_anomaly_scores else 0
                result = {
                    "logkey_anomaly": batch_logkey_anomaly,
                    "predicted_logkey": batch_predicted_logkey,
                    "anomaly_score": avg_anomaly_score
                }
                test_results.append(result)

                if idx < 10 or idx % 1000 == 0:
                    print(f"{data_type} sample {idx}: {result}")

            return test_results, normal_errors

    def predict_unsupervised(self):
        model = self.model.to(self.device)
        checkpoint = torch.load(self.model_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        print(f'Loaded model from: {self.model_path}')

        with open(self.vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        test_normal_loader, _ = generate(self.output_dir, 'test_normal')
        test_abnormal_loader, _ = generate(self.output_dir, 'test_abnormal')

        scale = None
        if self.is_time:
            scale_path = os.path.join(self.save_dir, "scale.pkl")
            with open(scale_path, "rb") as f:
                scale = pickle.load(f)

        # Test the model
        start_time = time.time()
        test_normal_results, _ = self.unsupervised_helper(
            model, test_normal_loader, vocab, 'test_normal',
            scale=scale, min_len=self.min_len)

        test_abnormal_results, _ = self.unsupervised_helper(
            model, test_abnormal_loader, vocab, 'test_abnormal',
            scale=scale, min_len=self.min_len)

        # Save results
        normal_results_path = os.path.join(self.save_dir, "test_normal_results.pkl")
        abnormal_results_path = os.path.join(self.save_dir, "test_abnormal_results.pkl")

        print(f"Saving test normal results to {normal_results_path}")
        with open(normal_results_path, "wb") as f:
            pickle.dump(test_normal_results, f)

        print(f"Saving test abnormal results to {abnormal_results_path}")
        with open(abnormal_results_path, "wb") as f:
            pickle.dump(test_abnormal_results, f)

        # Find best threshold and evaluate
        threshold_range = np.arange(0, 1, 0.01)  # More granular threshold search
        TH, TP, TN, FP, FN, P, R, F1 = self.find_best_threshold(
            test_normal_results,
            test_abnormal_results,
            threshold_range=threshold_range
        )

        print('\n=== Best Threshold Evaluation ===')
        print(f'Best threshold: {TH:.4f}')
        print("Confusion Matrix:")
        print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
        print(f'Precision: {P:.3f}%, Recall: {R:.3f}%, F1-measure: {F1:.3f}%')

        elapsed_time = time.time() - start_time
        print(f'\nElapsed time: {elapsed_time:.2f} seconds')

    def predict_supervised(self):
        """Supervised prediction using anomaly scores"""
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print(f'Model loaded from: {self.model_path}')

        test_logs, test_labels = session_window(self.output_dir, datatype='test')
        test_dataset = log_dataset(
            logs=test_logs,
            labels=test_labels,
            seq=self.sequentials,
            quan=self.quantitatives,
            sem=self.semantics,
            param=self.parameters
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True
        )

        TP, FP, FN, TN = 0, 0, 0, 0
        for log, label in tqdm(test_loader, desc="Testing"):
            features = []
            for value in log.values():
                features.append(value.clone().to(self.device))

            _, anomaly_scores = model(*features)
            predictions = (anomaly_scores.squeeze() > 0.5).long().cpu().numpy()
            labels = label.cpu().numpy()

            TP += ((predictions == 1) & (labels == 1)).sum()
            FP += ((predictions == 1) & (labels == 0)).sum()
            FN += ((predictions == 0) & (labels == 1)).sum()
            TN += ((predictions == 0) & (labels == 0)).sum()

        # Compute metrics
        precision = 100 * TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = 100 * TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print('\n=== Supervised Evaluation ===')
        print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
        print(f'Precision: {precision:.3f}%, Recall: {recall:.3f}%, F1-measure: {f1:.3f}%')