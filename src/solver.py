import os
import signal

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from torchvision.utils import make_grid
from time import time
from tqdm.auto import tqdm

from src.model import RidgeDetector
from utils.helpers import take_items

# TODO: 
# Debug
# More metrics

EXIT = False

# def interrupt_handler(signum, frame):
#     global EXIT
    
#     EXIT = not EXIT

#     if EXIT:
#         print("\nResuming execution.")
#     else:
#         print("\nTerminating.")

# signal.signal(signal.SIGINT, interrupt_handler)

optimizers = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    #...
}

def save_checkpoint(ckpt_path, epoch, model, optim, loss=None, score=None):
    checkpoint = { 
        'epoch': epoch,
        'loss': loss,
        'score': score,
        'model_args': model.get_args(),
        'model_state': model.state_dict(),
        'optim_name': str(optim.__class__).split('.')[2],
        'optim_args': optim.defaults,
        'optim_state': optim.state_dict(),
    }
    file = f'{epoch}_ckpt.pth'
    path = os.path.join(ckpt_path, file)
    torch.save(checkpoint, path)

def load_checkpoint(ckpt_path):
    checkpoint = torch.load(ckpt_path)
    last_epoch = checkpoint["epoch"]
    last_loss = checkpoint["loss"]
    last_score = checkpoint["score"]
    
    # Architecture is reinstantiated based on args saved
    model = RidgeDetector(checkpoint['model_args'])
    # Recover state
    model.load_state_dict(checkpoint['model_state'])
    # Optimizer is reinstantiated based on args saved
    optim_name = "adam" # checkpoint['optim_name']
    optim = optimizers[optim_name](filter(lambda p: p.requires_grad, model.parameters()), **checkpoint['optim_args'])
    # Recover state
    optim.load_state_dict(checkpoint['optim_state'])

    return last_epoch, last_loss, last_score, model, optim

def class_balanced_bce(logits, targets):
    total = np.prod(targets.shape[2:4])                      # total pixel count: scalar
    pos_count = torch.sum(targets, dim=(2,3), keepdim=True)  # + count per image: Bx6x1x1
    neg_count = total - pos_count                            # - count per image: Bx6x1x1
    pos_weights = targets * (neg_count / pos_count)          # b / (1 - b) term:  Bx6xHxW
    weights = torch.ones_like(targets) * (pos_count / total) # (1 - b) term:      Bx6xHxW

    # BCE with Logits == Sigmoid(Logits) + Weighted BCE:
    # weights * [pos_weights * y * -log(sigmoid(logits)) + (1 - y) * -log(1 - sigmoid(x))]
    loss = F.binary_cross_entropy_with_logits(logits, targets, 
                                              weight=weights, 
                                              pos_weight=pos_weights, 
                                              reduction='mean')
    print(logits.max()) # FIXME: NaN loss
    print(logits.min())

    return loss

def f1_score(logits, targets, threshold=0.5):
    preds = torch.sigmoid(logits) > threshold
    
    pred_edge = preds.bool()
    pred_non_edge = ~pred_edge
    gt_edge = targets.bool()
    gt_non_edge = ~gt_edge
    
    tp = (pred_edge & gt_edge).sum(dim=(2,3))     # Pred: Edge,     Target: Edge
    fp = (pred_edge & gt_non_edge).sum(dim=(2,3)) # Pred: Edge,     Target: Non-Edge
    fn = (pred_non_edge & gt_edge).sum(dim=(2,3)) # Pred: Non-Edge, Target: Edge
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    # tn = (pred_non_edge & gt_non_edge).sum(dim=(2,3)) # Pred: Non-Edge, Target: Non-Edge
    # total = np.prod(preds.shape[2:4])
    # tn = total - (tp + fp + fn)
    # acc = (tp + tn) / (tp + fp + tn + fn)

    f1[f1 != f1] = 0 # nan -> 0 TODO: Change when not needed

    return f1.mean()

unnormalize_imagenet = T.Compose([T.Normalize(mean=[ 0.,0.,0.], std=[1/0.229,1/0.224,1/0.225 ]),
                                  T.Normalize(mean=[-0.485,-0.456,-0.406], std=[1.,1.,1.])])

def revert_input_transforms(X, input_format):
    X_lst = []
    if input_format in ("darr", "rgb", "drgb"):
        X_lst.append(unnormalize_imagenet(X))
    
    elif input_format in ("rgb-darr", "rgb-drgb"):
        X1, X2 = torch.split(X, 3, dim=1)
        X_lst.extend([unnormalize_imagenet(X1), unnormalize_imagenet(X2)])

    return X_lst

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
        self.writers = {
            "Train": SummaryWriter(os.path.join(solver_args["log_path"], "train")),
            "Validation": SummaryWriter(os.path.join(solver_args["log_path"], "val")),
            "Test": SummaryWriter(os.path.join(solver_args["log_path"], "test")),
        }
        self.clear_history()
        self.solver_args = solver_args

    def get_args(self):
        return self.solver_args

    def clear_history(self):
        self.train_loss_hist = []
        self.train_score_hist = []
        self.val_loss_hist = []
        self.val_score_hist = []

    def forward(self, model, X, targets):
        # X: Bx3xHxW | Bx4xHxW | Bx6xHxW
        # logits: Bx6xHxW
        # targets: Bx1xHxW
        # output: Bx1xHxW
        logits = model(X)
        loss = self.loss_func(logits, targets.expand_as(logits)) # Single loss value (averaged)
        # logits[:,0:5,:,:].mean(dim=1, keepdims=True)
        # logits.mean(dim=1, keepdims=True)
        score = self.metric_func(logits[:,5,:,:], targets) # Single metric score (averaged)

        return logits, loss, score

    def backward(self, optim, loss):
        loss.backward()
        optim.step()
        optim.zero_grad()

    def run_one_epoch(self, epoch, loader, model, optim=None, args={}):
        mode = ""
        log_freq = 0
        vis_freq = 0
        if model.training:
            mode = "Train"
            log_freq = args.get('train_log_freq', 0)
            vis_freq = args.get('train_vis_freq', 0)
        else:
            mode = "Validation"
            log_freq = args.get('val_log_freq', 0)
            vis_freq = args.get('val_vis_freq', 0)

        input_format = args.get('input_format', 'darr')
        max_vis = args.get('max_vis', 5)

        writer = self.writers[mode]
        total_iter = len(loader)
        offset = (epoch-1) * total_iter
        losses = []
        scores = []
        
        # iter: [1, total_iter]
        for iter, batch in enumerate(tqdm(loader), 1):
            X = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            logits, loss, score = self.forward(model, X, targets)
            if model.training:
                self.backward(optim, loss)

            loss = loss.item()   # Tensor -> float
            score = score.item() # Tensor -> float
            losses.append(loss)
            scores.append(score)

            global_iter = iter + offset
            writer.add_scalar(f'Batch/Loss/', loss, global_iter)
            writer.add_scalar(f'Batch/Score/', score, global_iter)
            writer.flush()

            if log_freq > 0 and iter % log_freq == 0:
                print(f"[Iteration {iter}/{total_iter}] {mode} loss/score: {loss:.5f}/{score:.5f}")

            if vis_freq > 0 and iter % vis_freq == 0:
                preds = torch.sigmoid(logits)
                X_lst = revert_input_transforms(X, input_format)
                img_grid = prepare_batch_visualization([*X_lst, preds, targets], max_items=max_vis)
                writer.add_image("Input/Prediction/Target", img_grid, global_step=global_iter)
                writer.flush()

    
        mean_loss = np.mean(losses)
        mean_score = np.mean(scores)

        return mean_loss, mean_score

    def train(self, model, optim, train_loader, val_loader, train_args):
        model.to(self.device)

        ckpt_path = train_args['ckpt_path']
        max_epochs = train_args.get('max_epochs', 10)
        val_freq = train_args.get('val_freq', 1)
        ckpt_freq = train_args.get('ckpt_freq', 0)
        
        best_loss = np.inf
        best_score = -1
        
        # epoch: [1, max_epochs]
        epoch = 0
        for epoch in range(1, max_epochs+1):
            # Perform a full pass over training data
            model.train()
            mean_train_loss, mean_train_score = self.run_one_epoch(epoch, train_loader, model, optim, train_args)

            print(f"[Epoch {epoch}/{max_epochs}] Train mean loss/score: {mean_train_loss:.5f}/{mean_train_score:.5f}")

            self.train_loss_hist.append(mean_train_loss)
            self.train_score_hist.append(mean_train_score)
            
            self.writers['Train'].add_scalar('Epoch/Loss/', mean_train_loss, epoch)
            self.writers['Train'].add_scalar('Epoch/Score/', mean_train_score, epoch)
            self.writers['Train'].flush()

            # Perform a full pass over validation data (according to specified period)
            mean_val_loss = np.inf
            mean_val_score = -1
            if val_freq > 0 and epoch % val_freq == 0:
                model.eval()
                with torch.no_grad():
                    mean_val_loss, mean_val_score = self.run_one_epoch(epoch, val_loader, model, args=train_args)
                    
                    print(f"[Epoch {epoch}/{max_epochs}] Validation mean loss/score: {mean_val_loss:.5f}/{mean_val_score:.5f}")

                    self.val_loss_hist.append(mean_val_loss)
                    self.val_score_hist.append(mean_val_score)

                    self.writers['Validation'].add_scalar('Epoch/Loss/', mean_val_loss, epoch)
                    self.writers['Validation'].add_scalar('Epoch/Score/', mean_val_score, epoch)
                    self.writers['Validation'].flush()

            # Checkpoint is disabled for ckpt_freq <= 0
            make_ckpt = False
            if ckpt_freq > 0: 
                make_ckpt |= (epoch % ckpt_freq == 0) # Check for frequency
                make_ckpt |= (mean_val_loss < best_loss and mean_val_score > best_score) # Check for val loss & metric score improvement

            # If checkpointing is enabled and save checkpoint periodically or whenever there is an improvement 
            if make_ckpt:
                save_checkpoint(ckpt_path, epoch, model, optim, mean_val_loss, mean_val_score)

            # When KeyboardInterrupt is triggered, perform a controlled termination
            if EXIT:
                break

    def test(self, model, test_loader, test_args):
        model.to(self.device)
        model.eval()

        input_format = test_args.get('input_format', 'darr')
        max_vis = test_args.get('max_vis', 5)
        results = []

        with torch.no_grad():
            for iter, batch in enumerate(tqdm(test_loader), 1):
                X = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                logits = model(X)
                output = logits.mean(dim=1, keepdims=True) # TODO: Try out different stuff here, currently: mean(sideouts, fusion)
                preds = torch.sigmoid(output)
                
                X_lst = revert_input_transforms(X, input_format)
                img_grid = prepare_batch_visualization([*X_lst, preds, targets], max_items=max_vis)
                self.writers['Test'].add_image("Input/Prediction/Target", img_grid, global_step=iter)
                self.writers['Test'].flush()

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