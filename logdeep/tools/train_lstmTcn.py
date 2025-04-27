#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import os
import sys
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append('../../')
from logdeep.dataset.log import log_dataset
from logdeep.dataset.sample import sliding_window, session_window, split_features
from logdeep.tools.utils import save_parameters


class EnhancedTrainer():
    def __init__(self, model, options):
        # Initialize configuration
        self.model_name = options['model_name']
        self.save_dir = options['save_dir']
        self.output_dir = options['output_dir']
        self.window_size = options['window_size']
        self.batch_size = options['batch_size']
        self.device = options['device']
        self.lr_step = options['lr_step']
        self.lr_decay_ratio = options['lr_decay_ratio']
        self.accumulation_step = options['accumulation_step']
        self.max_epoch = options['max_epoch']
        self.n_epochs_stop = options["n_epochs_stop"]
        self.train_ratio = options['train_ratio']
        self.valid_ratio = options['valid_ratio']
        self.min_len = options["min_len"]
        self.pos_weight = options['pos_weight']

        # Feature configuration
        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.parameters = options['parameters']
        self.sample = options['sample']
        self.feature_num = options['feature_num']
        self.num_classes = options['num_classes']
        self.is_logkey = options["is_logkey"]
        self.is_time = options["is_time"]
        self.vocab_path = options["vocab_path"]

        # Initialize training state
        self.early_stopping = False
        self.epochs_no_improve = 0
        self.best_loss = float('inf')
        self.best_score = -1
        self.start_epoch = 0

        # Create directories
        os.makedirs(self.save_dir, exist_ok=True)
        scale_path = self.save_dir + "scale.pkl"
        if not os.path.exists(scale_path):
            open(scale_path, 'w').close()

        # Load and prepare data
        self._prepare_datasets()

        # Initialize model and optimizer
        self.model = model.to(self.device)
        self._init_optimizer(options)
        self._init_loss_functions()

        # Initialize logging
        self.log = {
            "train": {key: [] for key in ["epoch", "lr", "time", "loss", "next_loss", "anomaly_loss"]},
            "valid": {key: [] for key in ["epoch", "lr", "time", "loss", "next_loss", "anomaly_loss"]}
        }
        save_parameters(options, self.save_dir + "parameters.txt")

    def _prepare_datasets(self):
        print("Loading train dataset\n")
        logkeys, times = split_features(
            self.output_dir + "train",
            self.train_ratio,
            scale=None,
            scale_path=self.save_dir + "scale.pkl",
            min_len=self.min_len
        )

        train_logkeys, valid_logkeys, train_times, valid_times = train_test_split(
            logkeys, times, test_size=self.valid_ratio
        )

        print("Loading vocab")
        with open(self.vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        train_logs, train_labels = sliding_window(
            (train_logkeys, train_times),
            vocab=vocab,
            window_size=self.window_size,
        )

        val_logs, val_labels = sliding_window(
            (valid_logkeys, valid_times),
            vocab=vocab,
            window_size=self.window_size,
        )

        del train_logkeys, train_times, valid_logkeys, valid_times, vocab
        gc.collect()

        train_dataset = log_dataset(
            logs=train_logs,
            labels=train_labels,
            seq=self.sequentials,
            quan=self.quantitatives,
            sem=self.semantics,
            param=self.parameters
        )
        valid_dataset = log_dataset(
            logs=val_logs,
            labels=val_labels,
            seq=self.sequentials,
            quan=self.quantitatives,
            sem=self.semantics,
            param=self.parameters
        )

        del train_logs, val_logs
        gc.collect()

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True
        )
        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True
        )

        self.num_train_log = len(train_dataset)
        self.num_valid_log = len(valid_dataset)
        print(f'Find {self.num_train_log} train logs, {self.num_valid_log} validation logs')

    def _init_optimizer(self, options):
        if options['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=options['lr'],
                momentum=0.9,
                weight_decay=options['weight_decay']
            )
        elif options['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=options['lr'],
                betas=(0.9, 0.999),
                weight_decay=options['weight_decay']
            )
        else:
            raise NotImplementedError

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=list(options['lr_step']),
            gamma=options['lr_decay_ratio']
        )

    def _init_loss_functions(self):
        self.criterion_next = nn.CrossEntropyLoss(ignore_index=0)
        self.criterion_anomaly = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.pos_weight]).to(self.device)
        )

    def train(self, epoch):
        self.model.train()
        self.log['train']['epoch'].append(epoch)
        start = time.strftime("%H:%M:%S")
        lr = self.optimizer.param_groups[0]['lr']
        print(f"\nStarting epoch: {epoch} | phase: train | ⏰: {start} | Learning rate: {lr:.6f}")

        self.log['train']['lr'].append(lr)
        self.log['train']['time'].append(start)
        self.optimizer.zero_grad()

        total_loss = 0
        total_next_loss = 0
        total_anomaly_loss = 0
        num_batches = len(self.train_loader)

        with tqdm(self.train_loader, unit="batch") as tepoch:
            for i, (log, label) in enumerate(tepoch):
                features = [v.clone().detach().to(self.device) for v in log.values()]
                label = label.view(-1).to(self.device)

                # Forward pass
                next_pred, anomaly_score = self.model(*features)

                # Calculate losses
                next_loss = self.criterion_next(next_pred, label)
                anomaly_loss = self.criterion_anomaly(
                    anomaly_score.squeeze(),
                    (label != 0).float()  # Using non-padding as proxy for anomalies
                )
                loss = next_loss + 2.0 * anomaly_loss  # Weighted combination

                # Backward pass
                loss.backward()
                if (i + 1) % self.accumulation_step == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Update statistics
                total_loss += loss.item()
                total_next_loss += next_loss.item()
                total_anomaly_loss += anomaly_loss.item()

                tepoch.set_postfix({
                    'loss': f"{total_loss / (i + 1):.4f}",
                    'next': f"{total_next_loss / (i + 1):.4f}",
                    'anomaly': f"{total_anomaly_loss / (i + 1):.4f}"
                })

        # Log epoch statistics
        avg_loss = total_loss / num_batches
        avg_next = total_next_loss / num_batches
        avg_anomaly = total_anomaly_loss / num_batches

        self.log['train']['loss'].append(avg_loss)
        self.log['train']['next_loss'].append(avg_next)
        self.log['train']['anomaly_loss'].append(avg_anomaly)

        torch.cuda.empty_cache()
        self.scheduler.step()

    def valid(self, epoch):
        self.model.eval()
        self.log['valid']['epoch'].append(epoch)
        lr = self.optimizer.param_groups[0]['lr']
        self.log['valid']['lr'].append(lr)
        start = time.strftime("%H:%M:%S")
        print(f"\nStarting epoch: {epoch} | phase: valid | ⏰: {start}")
        self.log['valid']['time'].append(start)

        total_loss = 0
        total_next_loss = 0
        total_anomaly_loss = 0
        num_batches = len(self.valid_loader)

        all_probs = []
        all_labels = []
        anomaly_scores = []

        with torch.no_grad():
            with tqdm(self.valid_loader, unit="batch") as tepoch:
                for i, (log, label) in enumerate(tepoch):
                    features = [v.clone().detach().to(self.device) for v in log.values()]
                    label = label.view(-1).to(self.device)

                    # Forward pass
                    next_pred, anomaly_score = self.model(*features)

                    # Calculate losses
                    next_loss = self.criterion_next(next_pred, label)
                    anomaly_loss = self.criterion_anomaly(
                        anomaly_score.squeeze(),
                        (label != 0).float()
                    )
                    loss = next_loss + 2.0 * anomaly_loss

                    # Update statistics
                    total_loss += loss.item()
                    total_next_loss += next_loss.item()
                    total_anomaly_loss += anomaly_loss.item()

                    # Store predictions
                    probs = torch.softmax(next_pred, dim=-1)
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(label.cpu().numpy())
                    anomaly_scores.extend(anomaly_score.squeeze().cpu().numpy())

                    tepoch.set_postfix({
                        'loss': f"{total_loss / (i + 1):.4f}",
                        'next': f"{total_next_loss / (i + 1):.4f}",
                        'anomaly': f"{total_anomaly_loss / (i + 1):.4f}"
                    })

        # Log validation results
        avg_loss = total_loss / num_batches
        avg_next = total_next_loss / num_batches
        avg_anomaly = total_anomaly_loss / num_batches

        self.log['valid']['loss'].append(avg_loss)
        self.log['valid']['next_loss'].append(avg_next)
        self.log['valid']['anomaly_loss'].append(avg_anomaly)

        print(f"\nValidation loss: {avg_loss:.5f} (Next: {avg_next:.5f} | Anomaly: {avg_anomaly:.5f})")

        # Check for improvement
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.save_checkpoint(epoch, save_optimizer=True, suffix="bestloss")
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.n_epochs_stop:
                self.early_stopping = True
                print("Early stopping triggered")

    def save_checkpoint(self, epoch, save_optimizer=True, suffix=""):
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "best_loss": self.best_loss,
            "log": self.log,
            "best_score": self.best_score
        }
        if save_optimizer:
            checkpoint['optimizer'] = self.optimizer.state_dict()

        save_path = os.path.join(self.save_dir, f"{suffix}.pth")
        torch.save(checkpoint, save_path)
        print(f"Model checkpoint saved at {save_path}")

    def save_log(self):
        try:
            for key in ['train', 'valid']:
                pd.DataFrame(self.log[key]).to_csv(
                    f"{self.save_dir}{key}_log.csv",
                    index=False
                )
            print("Log saved")
        except Exception as e:
            print(f"Failed to save logs: {str(e)}")

    def plot_loss_curves(self):
        """Wrapper method to plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))

        # Plot total loss
        plt.subplot(2, 1, 1)
        plt.plot(self.log['train']['epoch'], self.log['train']['loss'], label='Train')
        plt.plot(self.log['valid']['epoch'], self.log['valid']['loss'], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # Plot component losses
        plt.subplot(2, 1, 2)
        plt.plot(self.log['train']['epoch'], self.log['train']['next_loss'], 'b--', label='Train Next')
        plt.plot(self.log['valid']['epoch'], self.log['valid']['next_loss'], 'g--', label='Valid Next')
        plt.plot(self.log['train']['epoch'], self.log['train']['anomaly_loss'], 'b:', label='Train Anomaly')
        plt.plot(self.log['valid']['epoch'], self.log['valid']['anomaly_loss'], 'g:', label='Valid Anomaly')
        plt.xlabel('Epoch')
        plt.ylabel('Component Losses')
        plt.title('Loss Components')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'loss_curves.png'))
        plt.close()

    def start_train(self):
        for epoch in range(self.start_epoch, self.max_epoch):
            if self.early_stopping:
                break
            self.train(epoch)
            self.valid(epoch)
            self.save_log()

        self.plot_loss_curves()  # Use the wrapper method
        print("Training completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='enhanced_lstm_tcn')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--accumulation_step', type=int, default=2)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--pos_weight', type=float, default=5.0)
    parser.add_argument('--resume_path', type=str, default=None)
    args = parser.parse_args()

    # Configuration dictionary
    options = {
        'model_name': args.model_name,
        'save_dir': "../output/bgl/enhanced_lstm_tcn/",
        'output_dir': "../output/bgl/",
        'vocab_path': "../output/bgl/vocab.pkl",
        'train_vocab': "../output/bgl/train",
        'window_size': 20,
        'batch_size': args.batch_size,
        'accumulation_step': args.accumulation_step,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'optimizer': 'adam',
        'lr': args.lr,
        'lr_step': (5, 10),
        'lr_decay_ratio': 0.5,
        'weight_decay': 1e-5,
        'max_epoch': args.max_epoch,
        'n_epochs_stop': 10,
        'train_ratio': 1.0,
        'valid_ratio': 0.1,
        'test_ratio': 1.0,
        'pos_weight': args.pos_weight,
        'min_len': 10,
        'sequentials': True,
        'quantitatives': False,
        'semantics': False,
        'parameters': False,
        'feature_num': 1,
        'num_classes': 200,
        'is_logkey': True,
        'is_time': False,
        'sample': 'sliding_window',
        'embedding_dim': 100,
        'hidden_size': 64,
        'num_layers': 4,
        'dropout': 0.3
    }

    # Initialize model (replace with your actual model import)
    model = EnhancedLstmTcn(
        input_size=options['feature_num'],
        hidden_size=options['hidden_size'],
        num_layers=options['num_layers'],
        vocab_size=options['num_classes'],
        embedding_dim=options['embedding_dim'],
        dropout=options['dropout']
    )

    # Initialize trainer
    trainer = EnhancedTrainer(model, options)

    # Resume training if path provided
    if args.resume_path is not None and os.path.exists(args.resume_path):
        checkpoint = torch.load(args.resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        trainer.start_epoch = checkpoint['epoch'] + 1
        trainer.best_loss = checkpoint['best_loss']
        trainer.log = checkpoint['log']
        print(f"Resumed training from epoch {trainer.start_epoch}")

    # Start training
    trainer.start_train()


if __name__ == "__main__":
    main()