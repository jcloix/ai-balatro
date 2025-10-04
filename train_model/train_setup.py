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


def prepare_training(heads, log_dir, lr=1e-4, patience=5, freeze_backbone=False, checkpoint=None):
    # Build the model
    model, device = build_model(heads)

    # Optionally freeze backbone layers, useful for smaller datasets
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    # Setup optimizer, scheduler, loss function, scaler, early stopping, and TensorBoard writer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    # Scheduler: StepLR for tiny dataset, ReduceLROnPlateau for medium dataset
    dataset_size = max(len(head.train_loader.dataset) for head in heads)
    if dataset_size is not None and dataset_size >= 100: # medium dataset
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    else:  # small dataset
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scaler = torch.amp.GradScaler(device="cuda") if torch.cuda.is_available() else None
    early_stopping = EarlyStopping(patience=patience)
    writer = SummaryWriter(log_dir=log_dir)

    # Load checkpoint if one has been provided
    start_epoch = 1
    if checkpoint:
        start_epoch = apply_checkpoint(checkpoint, model, optimizer, scheduler, scaler)
    return TrainingState(model, device, optimizer, scheduler, scaler, early_stopping, writer, start_epoch)