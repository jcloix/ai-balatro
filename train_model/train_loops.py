import torch
from train_model.metrics import top_k_accuracy, compute_confusion_matrix
from train_model.metrics import Metrics

# -----------------------------
# Training / Validation Steps
# -----------------------------

def forward_pass(model, images, labels, criterion, scaler=None):
    """
    Run forward pass and compute loss.
    Handles mixed precision if scaler is provided.
    Returns loss and model outputs.
    """
    if scaler: # Mixed precision training (faster and less memory on GPU)
        with torch.amp.autocast('cuda'): # Use mixed precision for this block
            outputs = model(images)
            loss = criterion(outputs, labels)
    else:
        outputs = model(images)
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


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    # Set model to training mode
    model.train()
    # Calculate training loss
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # Reset gradients
        _, loss = forward_pass(model, images, labels, criterion, scaler)
        backward_step(loss, optimizer, scaler)
        running_loss += loss.item() * images.size(0)

    # Determine dataset size safely
    dataset_size = len(loader.dataset) if hasattr(loader, "dataset") else len(loader) * images.size(0) if len(loader) > 0 else 0
    # Avoid division by zero
    return running_loss / dataset_size if dataset_size > 0 else 0.0


def validate(model, loader, criterion, device, compute_metrics=False, num_classes=None):
    # Set model to evaluation mode
    model.eval()

    # Calculate validation loss
    val_loss = 0.0

    # Accumulate outputs and labels if metrics are requested
    all_outputs, all_labels = [], []

    with torch.no_grad():  # Disable gradient computation for validation
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs, loss = forward_pass(model, images, labels, criterion)
            val_loss += loss.item() * images.size(0)

            if compute_metrics:  # Compute metrics if requested
                all_outputs.append(outputs)
                all_labels.append(labels)

    # Compute metrics if requested
    top3_acc = None
    cm = None
    if compute_metrics and all_outputs:  # Only compute if we have outputs
        all_outputs_tensor = torch.cat(all_outputs)
        all_labels_tensor = torch.cat(all_labels)
        # Compute top-k counters - to calculate how many times the guess was in the top-k answers
        top3_acc = top_k_accuracy(all_outputs_tensor, all_labels_tensor, k=3)
        # Compute confusion matrix - Instead of average, give class-by-class performance view
        cm = compute_confusion_matrix(model, loader, device, num_classes)

    # For empty loader, keep metrics None
    dataset_size = len(loader.dataset) if hasattr(loader, "dataset") else 0
    val_loss = val_loss / dataset_size if dataset_size > 0 else 0.0

    return Metrics.from_validation(val_loss, top3_acc, cm)

