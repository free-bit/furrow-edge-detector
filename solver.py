import os
import signal

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from time import time
from tqdm.auto import tqdm

# TODO: Checkpoint

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
    def __init__(self,
                 model,
                 optim_args):
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(optim_args["log_path"])
        self.ckpt_path = optim_args["ckpt_path"]
        self.clear_history()

        self.model.to(self.device)
        self.optim = None # torch.nn.optim(filter(lambda p: p.requires_grad, model.parameters()), **self.optim_args)
        self.loss_func = None
        self.acc_func = None

    def clear_history(self):
        self.train_loss_hist = []
        self.train_acc_hist = []
        self.val_loss_hist = []
        self.val_acc_hist = []
    
    def save_checkpoint(self, epoch):
        # TODO: Architectural parameters must also be saved
        checkpoint = { 
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optim.state_dict()}
        file = f'{epoch}_ckpt.pth'
        path = os.path.join(self.ckpt_path, file)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        # TODO: Architecture must be reinstantiated wrt earlier run
        # model = TheModelClass(*args, **kwargs)
        # checkpoint = torch.load(path)
        # model.load_state_dict(checkpoint["model"])
        # model.to(device)
        pass

    def forward(self, batch): # TODO: loss calculation, accuracy and so on
        # ... = batch
        # preds = self.model(...)

        # loss = self.loss_func(preds, y)
        # acc = self.acc_func(preds, y)

        # return loss, acc

        return None, None

    def backward(self, loss):
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()

    def train_one_epoch(self, train_loader, epoch, log_freq):
        self.model.train()

        total_iter = len(train_loader)
        train_losses = []
        train_accs = []
        
        # iter: [1, total_iter]
        for iter, batch in enumerate(tqdm(train_loader), 1):
            train_loss, train_acc = self.forward(batch)
            self.backward(train_loss)

            train_loss = train_loss.data.cpu().numpy()
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            self.writer.add_scalar('Batch/Loss/Train', train_loss, iter + (epoch-1) * total_iter)
            self.writer.add_scalar('Batch/Accuracy/Train', train_acc, iter + (epoch-1) * total_iter)
            self.writer.flush()

            if log_freq > 0 and iter % log_freq == 0:
                print(f"[Iteration {iter}/{total_iter}] TRAIN loss: {train_loss}")
    
        mean_train_loss = np.mean(train_losses)
        mean_train_acc = np.mean(train_accs)

        return mean_train_loss, mean_train_acc

    def validate_one_epoch(self, val_loader, epoch, log_freq=0):
        self.model.eval()

        with torch.no_grad():
            total_iter = len(val_loader)
            val_losses = []
            val_accs = []

            # iter: [1, total_iter]
            for iter, batch in enumerate(tqdm(val_loader), 1):
                val_loss, val_acc = self.forward(batch)
                val_loss = val_loss.data.cpu().numpy()
                val_losses.append(val_loss)
                val_accs.append(val_acc)

                self.writer.add_scalar('Batch/Loss/Val', val_loss, iter + (epoch-1) * total_iter)
                self.writer.add_scalar('Batch/Accuracy/Val', val_acc, iter + (epoch-1) * total_iter)
                self.writer.flush()

                if log_freq > 0 and iter % log_freq == 0:
                    print(f"[Iteration {iter}/{total_iter}] Val loss: {val_loss}")

            mean_val_loss = np.mean(val_losses)
            mean_val_acc = np.mean(val_accs)

        return mean_val_loss, mean_val_acc

    def train(self, train_loader, val_loader, max_epochs=10, val_freq=0, log_freq=0, ckpt_freq=0):
        # epoch: [1, max_epochs]
        for epoch in range(1, max_epochs+1):
            # Perform a full pass over training data
            mean_train_loss, mean_train_acc = self.train_one_epoch(train_loader, epoch, log_freq)

            print(f"[Epoch {epoch}/{max_epochs}] TRAIN mean acc/loss: {mean_train_acc}/{mean_train_loss}")

            self.train_loss_hist.append(mean_train_loss)
            self.train_acc_hist.append(mean_train_acc)
            
            self.writer.add_scalar('Epoch/Loss/Train', mean_train_loss, epoch)
            self.writer.add_scalar('Epoch/Accuracy/Train', mean_train_acc, epoch)
            self.writer.flush()

            # Perform a full pass over validation data
            if val_freq > 0 and iter % val_freq == 0:
                mean_val_loss, mean_val_acc = self.validate_one_epoch(val_loader, epoch, log_freq)
                
                print(f"[Epoch {epoch}/{max_epochs}] VAL mean acc/loss: {mean_val_acc}/{mean_val_loss}")

                self.val_loss_hist.append(mean_val_loss)
                self.val_acc_hist.append(mean_val_acc)

                self.writer.add_scalar('Epoch/Loss/Val', mean_val_loss, epoch)
                self.writer.add_scalar('Epoch/Accuracy/Val', mean_val_acc, epoch)
                self.writer.flush()

            # TODO: Save model with best validation loss or accuracy
            if ckpt_freq > 0 and iter % ckpt_freq == 0:
                self.save_checkpoint(epoch)

            # When KeyboardInterrupt is triggered, perform a controlled termination
            if EXIT:
                break

        # self.writer.add_hparams(self.hparam_dict, {
        #     'HParam/Accuracy/Val': self.val_acc_hist[-1],
        #     'HParam/Accuracy/Train': self.train_acc_hist[-1],
        #     'HParam/Loss/Val': self.val_loss_hist[-1],
        #     'HParam/Loss/Train': self.train_loss_hist[-1]
        # })
        # self.writer.flush()

    def test(self, test_loader):
        self.model.eval()
        
        outputs = []

        with torch.no_grad():
            for batch in tqdm(test_loader):
                output = self.model(batch)
                outputs.append(output)
        
        return outputs