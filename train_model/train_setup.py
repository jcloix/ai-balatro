#models.py
import torch
import os
from torch import nn, optim
from models.models import build_model
from torch.utils.tensorboard import SummaryWriter
from train_model.train_state import EarlyStopping, TrainingState


def prepare_training(num_classes, log_dir, lr=1e-4, patience=5, freeze_backbone=False, resume_path=None, dataset_size=None):
    # Build the model based on pretrained ResNet18 and replace final layer.
    model, device = build_model(num_classes=num_classes)

    # Optionally freeze backbone layers, useful for smaller datasets
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    # Setup optimizer, scheduler, loss function, scaler, early stopping, and TensorBoard writer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    # Scheduler: StepLR for tiny dataset, ReduceLROnPlateau for medium dataset
    if dataset_size is not None and dataset_size >= 100: # medium dataset
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    else:  # small dataset
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scaler = torch.amp.GradScaler(device="cuda") if torch.cuda.is_available() else None
    early_stopping = EarlyStopping(patience=patience)
    writer = SummaryWriter(log_dir=log_dir)

    # Load checkpoint if resume_path provided
    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"[INFO] Resumed training from {resume_path}")

    return TrainingState(model, device, optimizer, scheduler, criterion, scaler, early_stopping, writer)
