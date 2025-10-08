#train_loops.py
import torch
import contextlib
from models.models import unwrap_model

# -----------------------------
# Training / Validation Steps
# -----------------------------

def forward_pass(model, images, labels, criterion, scaler=None, task_name=None):
    """
    Run forward pass and compute loss.
    Handles mixed precision if scaler is provided.
    Returns loss and model outputs.
    Supports models that return a dict (e.g. with 'logits' or 'out').
    """
    ctx = torch.amp.autocast("cuda") if scaler else contextlib.nullcontext()
    with ctx:
        outputs = unwrap_model(model, images, task_name)
        loss = criterion(outputs, labels)
    return outputs, loss


def backward_step(loss, optimizer, scaler=None):
    """
    Backward pass and optimizer step.
    Handles mixed precision scaling if scaler is provided.
    """
    if scaler: # Mixed precision training (faster and less memory on GPU)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()


def train_one_epoch(head, state, epoch_res):
    loader = head.train_loader
    # Set model to training mode
    state.model.train()
    # Calculate training loss
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(state.device), labels.to(state.device)
        state.optimizer.zero_grad()  # Reset gradients
        _, loss = forward_pass(state.model, images, labels, head.criterion, state.scaler, head.name)
        backward_step(loss, state.optimizer, state.scaler)
        running_loss += loss.item() * images.size(0)

    # Determine dataset size safely
    dataset_size = len(loader.dataset) if hasattr(loader, "dataset") else len(loader) * images.size(0) if len(loader) > 0 else 0
    # Avoid division by zero
    train_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    epoch_res.set_train(train_loss=train_loss)
    return train_loss


def validate(head, state, epoch_res):
    loader = head.val_loader
    # Set model to evaluation mode
    state.model.eval()

    # Calculate validation loss
    val_loss = 0.0

    # Accumulate outputs and labels if metrics are requested
    all_outputs, all_labels = [], []

    with torch.no_grad():  # Disable gradient computation for validation
        for images, labels in loader:
            images, labels = images.to(state.device), labels.to(state.device)
            outputs, loss = forward_pass(state.model, images, labels, head.criterion, scaler=None, task_name=head.name)
            val_loss += loss.item() * images.size(0)

            if head.metrics:  # Compute metrics if requested
                all_outputs.append(outputs)
                all_labels.append(labels)

    # For empty loader, keep metrics None
    dataset_size = len(loader.dataset) if hasattr(loader, "dataset") else 0
    val_loss = val_loss / dataset_size if dataset_size > 0 else 0.0

    epoch_res.set_val(val_loss=val_loss,val_outputs=all_outputs,val_labels=all_labels)
    return val_loss


