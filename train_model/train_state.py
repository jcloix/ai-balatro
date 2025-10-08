
#train_state.py
from typing import Optional
import torch
# -----------------------------
# TrainingState
# -----------------------------
class TrainingState:
    """
    Container for model, device, optimizer, scheduler, criterion, scaler, and other training parameters.
    """
    def __init__(self, model, device, optimizer, scheduler, scaler, early_stopping, writer, start_epoch = 0):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.early_stopping = early_stopping
        self.writer = writer
        self.best_val_loss = float('inf')
        self.start_epoch = start_epoch

# -----------------------------
# Early Stopping
# -----------------------------
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# -----------------------------
# Epoch result
# -----------------------------
class EpochResult:
    """
    Container for results of a single epoch.
    Can be filled incrementally: first with epoch, then with train/val results.
    """

    def __init__(self, epoch: int):
        self.epoch: int = epoch
        self.train_loss: Optional[float] = None
        self.val_loss: Optional[float] = None
        self.val_outputs: Optional[torch.Tensor] = None
        self.val_labels: Optional[torch.Tensor] = None

    # -----------------------------
    # Fill train results
    # -----------------------------
    def set_train(self, train_loss: float):
        self.train_loss = train_loss

    # -----------------------------
    # Fill validation results
    # -----------------------------
    def set_val(self, val_loss: float, val_outputs=None, val_labels=None):
        self.val_loss = val_loss
        if val_outputs:
            self.val_outputs = torch.cat(val_outputs)
        if val_labels:
            self.val_labels = torch.cat(val_labels)

