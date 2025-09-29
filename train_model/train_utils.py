import torch
from torch import nn, optim
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from train_model.dataset import get_train_val_loaders, CardDataset, load_merged_labels
from train_model.train_config import Config
from config.config import LABELS_FILE, AUGMENTED_LABELS_FILE, MERGED_LABELS_FILE

class TrainingState:
    """
    Container for model, device, optimizer, scheduler, criterion, scaler, and other training parameters.
    """
    def __init__(self, model, device, optimizer, scheduler, criterion, scaler, early_stopping, writer):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.scaler = scaler
        self.early_stopping = early_stopping
        self.writer = writer
        self.best_val_loss = float('inf')

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
# Model Setup
# -----------------------------
def build_model(num_classes):
    # Use a pre-trained ResNet18 (convolutional neural network) model and modify the final layer
    model = models.resnet18(pretrained=True)
    # Create a linear layer with the appropriate number of output classes for our dataset
    # This replaces only the FC layer at the end of the model, so that we can leverage the pre-trained weights
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    # Send model to device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), device

def prepare_training(num_classes, log_dir, lr=1e-4, patience=5):
    # Build the model based on pretrained ResNet18 and replace final layer.
    model, device = build_model(num_classes=num_classes)
    # Setup optimizer, scheduler, loss function, scaler, early stopping, and TensorBoard writer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scaler = torch.amp.GradScaler(device="cuda") if torch.cuda.is_available() else None
    early_stopping = EarlyStopping(patience=patience)
    writer = SummaryWriter(log_dir=log_dir)

    return TrainingState(
        model=model,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        scaler=scaler,
        early_stopping=early_stopping,
        writer=writer
    )
