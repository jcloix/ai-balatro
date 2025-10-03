
#train_state.py
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
