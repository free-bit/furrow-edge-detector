import os
import signal

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from time import time
from tqdm.auto import tqdm

from src.dataloader import FurrowDataset
from utils.helpers import save_checkpoint, take_items

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

def class_balanced_bce(logits, targets):
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

def f1_score(logits, targets, threshold=0.5):
    preds = torch.sigmoid(logits) > threshold
    
    tp = (preds.bool() * targets.bool()).sum(dim=(2,3))
    fp = (preds.bool() * ~targets.bool()).sum(dim=(2,3))
    tn = (~preds.bool() * ~targets.bool()).sum(dim=(2,3))
    
    precision = tp / (tp + fp)
    recall = tp / (tp + tn)
    f1 = (2 * precision * recall) / (precision + recall)

    f1[f1 != f1] = 0 # nan -> 0 TODO: Change when not needed

    return f1.mean()

def prepare_batch_visualization(batch_groups, start=0, end=np.inf, max_items=3):
    """Construct a grid from batch_groups (type: list). 
    
    Each individual batch group is a batch of images (type: tensor) is with shape: B x C* x H x W.
    Order of individual batch group in batch_groups defines the order of images in columns.
    Number of rows in the grid depends on number of images to take from each group (= max_items).
    Number of columns in the grid depends on the size of batch_groups (= num_groups).
    *Dimension C might be different for each batch in batch_groups. Smaller channels are promoted to maximum number of channels (= C_max).

    Parameters
    ----------
    batch_groups : list 
        Different groups of image batches e.g. [rgb_batch, darr_batch, pred_batch, mask_batch...]
    start : int 
        Start index for range
    end : int
        End index for range
    max_items : int 
        Number of items to take from range: [start, end]

    Returns 
    -------
    img_grid : tensor 
        Resulting grid with shape: (C_max) x (max_items * H + padding) x (num_groups * W + padding)
    """
    with torch.no_grad():
        num_batch_groups = len(batch_groups)

        # Determine C_max, send tensors to cpu while detaching from graph
        C_max, shape = -1, None
        for i in range(num_batch_groups):
            shape = batch_groups[i].shape
            C_max = shape[1] if shape[1] > C_max else C_max
            batch_groups[i] = batch_groups[i].detach().cpu()
        
        # Take max_items from range: [start, end]
        shape = (-1, C_max, *shape[2:4])
        for i in range(num_batch_groups):
            batch_groups[i] = torch.stack(take_items(batch_groups[i], start, end, max_items), dim=0)
            batch_groups[i] = batch_groups[i].expand(shape) # Promote channels to the C_max

        # Construct stack of interleaving batch_groups: (2*B)xCxHxW
        zipped = torch.stack(batch_groups, dim=1)
        zipped = torch.reshape(zipped, (-1, *shape[1:4]))

        # Construct a row taking one image from each image set
        img_grid = make_grid(zipped, nrow=num_batch_groups, normalize=True, scale_each=True)

        return img_grid

LOSSES = {
    "bce": F.binary_cross_entropy_with_logits, 
    "class_balanced_bce": class_balanced_bce,
}

METRICS = {
    "f1": f1_score
}

class Solver(object):
    def __init__(self, solver_args):
        loss_id = solver_args["loss_func"]
        metric_id = solver_args["metric_func"]
        self.loss_func = LOSSES[loss_id]
        self.metric_func = METRICS[metric_id]
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

    def forward(self, model, X, targets):
        # X: Bx1xHxW | Bx3xHxW | Bx4xHxW | Bx6xHxW
        # targets: Bx1xHxW
        logits = model(X)
        # preds = torch.sigmoid(logits)
        loss = self.loss_func(logits, targets)     # Single loss value (averaged over batch)
        metric = self.metric_func(logits, targets) # Single metric score (averaged over batch)

        return logits, loss, metric

    def backward(self, optim, loss):
        loss.backward()
        optim.step()
        optim.zero_grad()

    def run_one_epoch(self, epoch, loader, model, 
                      optim=None, 
                      log_freq=0, 
                      vis_freq=0, 
                      max_vis=5,
                      input_format='darr'):
        mode = "Train" if model.training else 'Validation'

        total_iter = len(loader)
        offset = (epoch-1) * total_iter
        losses = []
        metrics = []
        
        # iter: [1, total_iter]
        for iter, batch in enumerate(tqdm(loader), 1):
            samples = FurrowDataset.split_item(batch, input_format=input_format)
            X = samples['input'].to(self.device)
            targets = samples['gt'].to(self.device)
            logits, loss, metric = self.forward(model, X, targets)
            if model.training:
                self.backward(optim, loss)

            loss = loss.item()     # Tensor -> float
            metric = metric.item() # Tensor -> float
            losses.append(loss)
            metrics.append(metric)

            global_iter = iter + offset
            self.writer.add_scalar(f'Batch/Loss/{mode}', loss, global_iter)
            self.writer.add_scalar(f'Batch/Metric/{mode}', metric, global_iter)
            self.writer.flush()

            if log_freq > 0 and iter % log_freq == 0:
                print(f"[Iteration {iter}/{total_iter}] {mode} loss/metric: {loss:.5f}/{metric:.5f}")

            if vis_freq > 0 and iter % vis_freq == 0:
                preds = torch.sigmoid(logits)
                img_grid = prepare_batch_visualization([samples['input'], preds, targets], max_items=max_vis)
                self.writer.add_image(f"{mode}", img_grid, global_step=global_iter)
                self.writer.flush()

    
        mean_loss = np.mean(losses)
        mean_metric = np.mean(metrics)

        return mean_loss, mean_metric

    def train(self, model, optim, train_loader, val_loader, train_args):
        model.to(self.device)

        ckpt_path = train_args['ckpt_path']
        max_epochs = train_args.get('max_epochs', 10)
        val_freq = train_args.get('val_freq', 1)
        log_freq = train_args.get('log_freq', 0)
        vis_freq = train_args.get('vis_freq', 0)
        ckpt_freq = train_args.get('ckpt_freq', 0)
        input_format = train_args.get('input_format', 'darr')
        
        best_loss = np.inf
        best_metric = -1
        
        # epoch: [1, max_epochs]
        epoch = 0
        for epoch in range(1, max_epochs+1):
            # Perform a full pass over training data
            model.train()
            mean_train_loss, mean_train_metric = self.run_one_epoch(epoch, train_loader, model, optim, 
                                                                    log_freq=log_freq, vis_freq=vis_freq, 
                                                                    input_format=input_format)

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
                model.eval()
                with torch.no_grad():
                    mean_val_loss, mean_val_metric = self.run_one_epoch(epoch, val_loader, model, 
                                                                        log_freq=log_freq, vis_freq=vis_freq, 
                                                                        input_format=input_format)
                    
                    print(f"[Epoch {epoch}/{max_epochs}] Validation mean loss/metric: {mean_val_loss:.5f}/{mean_val_metric:.5f}")

                    self.val_loss_hist.append(mean_val_loss)
                    self.val_metric_hist.append(mean_val_metric)

                    self.writer.add_scalar('Epoch/Loss/Validation', mean_val_loss, epoch)
                    self.writer.add_scalar('Epoch/Metric/Validation', mean_val_metric, epoch)
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

    def test(self, model, test_loader, test_args):
        model.to(self.device)
        model.eval()

        input_format = test_args.get('input_format', 'darr')
        results = []

        with torch.no_grad():
            for iter, batch in enumerate(tqdm(test_loader), 1):
                samples = FurrowDataset.split_item(batch, input_format)
                X = samples['input'].to(self.device)
                targets = samples['gt'].to(self.device)
                logits = model(X)
                preds = torch.sigmoid(logits)
                
                img_grid = prepare_batch_visualization([samples['input'], preds, targets], max_items=np.inf)
                self.writer.add_image("Test", img_grid, global_step=iter)
                self.writer.flush()

                results.append(preds)
        
        return results

    def __str__(self):
        device = self.solver_args["device"]
        loss_func = self.solver_args["loss_func"]
        metric_func = self.solver_args["metric_func"]
        log_path = self.solver_args["log_path"]
        info = "Status of solver:\n"+\
               f"* Device: {device}\n"+\
               f"* Loss function: {loss_func}\n"+\
               f"* Metric function: {metric_func}\n"+\
               f"* Log path: {log_path}\n"
        return info