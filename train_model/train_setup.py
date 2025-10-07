#train_setup.py
import torch
from torch import optim
from models.models import build_multi_model
from torch.utils.tensorboard import SummaryWriter
from train_model.train_state import EarlyStopping, TrainingState
from train_model.persistence import apply_checkpoint


def build_model(heads):
    # Build the model
    head_configs = {head.name: head.num_classes for head in heads}
    return build_multi_model(head_configs)


def prepare_training(heads, strategy, log_dir, patience=5, checkpoint=None):
    # Build the model
    model, device = build_model(heads)

    # Appply the strategies
    model, optimizer, scheduler = strategy.apply(model)

    # Init criterion for each heads
    for head in heads:
        head.init_criterion(model, device)

    scaler = torch.amp.GradScaler(device="cuda") if torch.cuda.is_available() else None
    early_stopping = EarlyStopping(patience=patience)
    writer = SummaryWriter(log_dir=log_dir)

    # Load checkpoint if one has been provided
    start_epoch = 1
    if checkpoint:
        start_epoch = apply_checkpoint(checkpoint, model, optimizer, scheduler, scaler)
    return TrainingState(model, device, optimizer, scheduler, scaler, early_stopping, writer, start_epoch)