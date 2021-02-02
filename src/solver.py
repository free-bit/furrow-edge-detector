import os
import signal

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from time import time
from tqdm.auto import tqdm

from src.dataloader import FurrowDataset
from utils.helpers import save_checkpoint, load_checkpoint

# TODO: 
# Debug
# More metrics

EXIT = False

def interrupt_handler(signum, frame):
    global EXIT
    
    EXIT = not EXIT

    if EXIT:
        print("\nResuming execution.")
    else:
        print("\nTerminating.")

signal.signal(signal.SIGINT, interrupt_handler)

class Solver(object):
    def __init__(self, solver_args):
        # self.loss_func = solver_args["loss_func"]
        # self.metric_func = solver_args["metric_func"]
        self.device = solver_args["device"]
        self.writer = SummaryWriter(solver_args["log_path"])
        self.clear_history()
        self.solver_args = solver_args

    def get_args(self):
        return self.solver_args

    def clear_history(self):
        self.train_loss_hist = []
        self.train_metric_hist = []
        self.val_loss_hist = []
        self.val_metric_hist = []

    def class_balanced_bce(self, logits, targets):
        total = np.prod(targets.shape[2:4])                      # total pixel count: scalar
        pos_count = torch.sum(targets, dim=(2,3), keepdim=True)  # + count per image: Nx1x1x1
        neg_count = total - pos_count                            # - count per image: Nx1x1x1
        pos_weights = targets * (neg_count / pos_count)          # b / (1 - b) term:  Nx1xHxW
        weights = torch.ones_like(targets) * (pos_count / total) # (1 - b) term:      Nx1xHxW

        # BCE with Logits == Sigmoid(Logits) + Weighted BCE:
        # weights * [pos_weights * y * -log(sigmoid(logits)) + (1 - y) * -log(1 - sigmoid(x))]
        # 'mean' reduction does the followings:
        # 1) avg loss of pixels for each image
        # 2) avg loss of images in the batch
        loss = F.binary_cross_entropy_with_logits(logits, targets, 
                                                  weight=weights, 
                                                  pos_weight=pos_weights, 
                                                  reduction='mean')

        return loss

    def f1_score(self, logits, targets, threshold=0.5):
        preds = torch.sigmoid(logits) > threshold
        
        tp = (preds.bool() * targets.bool()).sum(dim=(2,3))
        fp = (preds.bool() * ~targets.bool()).sum(dim=(2,3))
        tn = (~preds.bool() * ~targets.bool()).sum(dim=(2,3))
        
        precision = tp / (tp + fp)
        recall = tp / (tp + tn)
        f1 = (2 * precision * recall) / (precision + recall)

        f1[f1 != f1] = 0 # nan -> 0 TODO: Change when not needed
    
        return f1.mean()

    # TODO: forward method has to adaptive select the inputs to pass (dataset args have to read here)
    def forward(self, model, batch):
        # depth_img: Nx3xHxW, depth_arr: Nx1xHxW, targets: Nx1xHxW
        depth_arr, edge_mask, rgb_img, depth_img = FurrowDataset.split_item(batch)
        X = depth_arr.to(self.device)
        # X = rgb_img.to(self.device)
        # X = depth_img.to(self.device)
        targets = edge_mask.to(self.device)
        logits = model(X)
        
        loss = self.class_balanced_bce(logits, targets) # Single loss value (averaged over batch)
        metric = self.f1_score(logits, targets)         # Single metric score (averaged over batch)

        return loss, metric

    def backward(self, optim, loss):
        loss.backward()
        optim.step()
        optim.zero_grad()

    def train_one_epoch(self, model, optim, train_loader, epoch, log_freq):
        model.train()

        total_iter = len(train_loader)
        train_losses = []
        train_metrics = []
        
        # iter: [1, total_iter]
        for iter, batch in enumerate(tqdm(train_loader), 1):
            train_loss, train_metric = self.forward(model, batch)
            self.backward(optim, train_loss)

            train_loss = train_loss.item()     # Tensor -> float
            train_metric = train_metric.item() # Tensor -> float
            train_losses.append(train_loss)
            train_metrics.append(train_metric)

            self.writer.add_scalar('Batch/Loss/Train', train_loss, iter + (epoch-1) * total_iter)
            self.writer.add_scalar('Batch/Metric/Train', train_metric, iter + (epoch-1) * total_iter)
            self.writer.flush()

            if log_freq > 0 and iter % log_freq == 0:
                print(f"[Iteration {iter}/{total_iter}] Train loss/metric: {train_loss:.5f}/{train_metric:.5f}")
    
        mean_train_loss = np.mean(train_losses)
        mean_train_metric = np.mean(train_metrics)

        return mean_train_loss, mean_train_metric

    def validate_one_epoch(self, model, val_loader, epoch, log_freq=0):
        model.eval()

        with torch.no_grad():
            total_iter = len(val_loader)
            val_losses = []
            val_metrics = []

            # iter: [1, total_iter]
            for iter, batch in enumerate(tqdm(val_loader), 1):
                val_loss, val_metric = self.forward(model, batch)
                
                val_loss = val_loss.item()     # Tensor -> float
                val_metric = val_metric.item() # Tensor -> float
                val_losses.append(val_loss)
                val_metrics.append(val_metric)

                self.writer.add_scalar('Batch/Loss/Val', val_loss, iter + (epoch-1) * total_iter)
                self.writer.add_scalar('Batch/Metric/Val', val_metric, iter + (epoch-1) * total_iter)
                self.writer.flush()

                if log_freq > 0 and iter % log_freq == 0:
                    print(f"[Iteration {iter}/{total_iter}] Val loss/metric: {val_loss:.5f}/{val_metric:.5f}")

            mean_val_loss = np.mean(val_losses)
            mean_val_metric = np.mean(val_metrics)

        return mean_val_loss, mean_val_metric

    def train(self, model, optim, train_loader, val_loader, train_args):
        model.to(self.device)
        ckpt_path = train_args['ckpt_path']
        max_epochs = train_args.get('max_epochs', 10)
        val_freq = train_args.get('val_freq', 1)
        log_freq = train_args.get('log_freq', 0)
        ckpt_freq = train_args.get('ckpt_freq', 0)
        
        best_loss = np.inf
        best_metric = -1
        
        # epoch: [1, max_epochs]
        epoch = 0
        for epoch in range(1, max_epochs+1):
            # Perform a full pass over training data
            mean_train_loss, mean_train_metric = self.train_one_epoch(model, optim, train_loader, epoch, log_freq)

            print(f"[Epoch {epoch}/{max_epochs}] Train mean loss/metric: {mean_train_loss:.5f}/{mean_train_metric:.5f}")

            self.train_loss_hist.append(mean_train_loss)
            self.train_metric_hist.append(mean_train_metric)
            
            self.writer.add_scalar('Epoch/Loss/Train', mean_train_loss, epoch)
            self.writer.add_scalar('Epoch/Metric/Train', mean_train_metric, epoch)
            self.writer.flush()

            # Perform a full pass over validation data (according to specified period)
            mean_val_loss = np.inf
            mean_val_metric = -1
            if val_freq > 0 and epoch % val_freq == 0:
                mean_val_loss, mean_val_metric = self.validate_one_epoch(model, val_loader, epoch, log_freq)
                
                print(f"[Epoch {epoch}/{max_epochs}] Val mean loss/metric: {mean_val_loss:.5f}/{mean_val_metric:.5f}")

                self.val_loss_hist.append(mean_val_loss)
                self.val_metric_hist.append(mean_val_metric)

                self.writer.add_scalar('Epoch/Loss/Val', mean_val_loss, epoch)
                self.writer.add_scalar('Epoch/Metric/Val', mean_val_metric, epoch)
                self.writer.flush()

            # Checkpoint is disabled for ckpt_freq <= 0
            make_ckpt = False
            if ckpt_freq > 0: 
                make_ckpt |= (epoch % ckpt_freq == 0) # Check for frequency
                make_ckpt |= (mean_val_loss < best_loss and mean_val_metric > best_metric) # Check for val loss & metric score improvement

            # If checkpointing is enabled and save checkpoint periodically or whenever there is an improvement 
            if make_ckpt:
                save_checkpoint(ckpt_path, epoch, model, optim, mean_val_loss, mean_val_metric)

            # When KeyboardInterrupt is triggered, perform a controlled termination
            if EXIT:
                break

        # self.writer.add_hparams(self.hparam_dict, {
        #     'HParam/Metric/Val': self.val_metric_hist[-1],
        #     'HParam/Metric/Train': self.train_metric_hist[-1],
        #     'HParam/Loss/Val': self.val_loss_hist[-1],
        #     'HParam/Loss/Train': self.train_loss_hist[-1]
        # })
        # self.writer.flush()

    def test(self, model, test_loader):
        model.to(self.device)
        model.eval()
        
        outputs = []

        with torch.no_grad():
            for batch in tqdm(test_loader):
                output = model(batch)
                outputs.append(output)
        
        return outputs

    def __str__(self):
        device = self.solver_args["device"]
        log_path = self.solver_args["log_path"]
        info = "Status of solver:\n"+\
               f"* Device: {device}\n"+\
               f"* Log path: {log_path}\n"
        return info