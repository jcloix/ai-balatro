# train_model/train.py
import os
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from train_model.dataset import CardDataset, get_train_val_loaders, load_merged_labels
from train_model.train_config import Config
from config.config import MODELS_DIR, LABELS_FILE, AUGMENTED_LABELS_FILE, MERGED_LABELS_FILE
import argparse
from train_model.train_utils import build_model, EarlyStopping, TrainingState, prepare_training

# -----------------------------
# Argument Parser
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train Balatro card recognition model")
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE)
    parser.add_argument("--val-split", type=float, default=Config.VAL_SPLIT)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument(
        "--transforms",
        type=str,
        default="train",
        choices=list(Config.TRANSFORMS.keys())
    )
    parser.add_argument("--use-augmented", action="store_true")
    return parser.parse_args()


# -----------------------------
# DataLoader Setup
# -----------------------------
def load_dataloaders(batch_size, val_split, use_augmented=False):
    """
    Merge labels, create dataset, and return train/val loaders.
    """
    augmented_file = AUGMENTED_LABELS_FILE if use_augmented else None

    # Merge labels once and save snapshot
    merged_labels = load_merged_labels(LABELS_FILE, augmented_file, save_path=MERGED_LABELS_FILE)

    # Create dataset (no internal transforms)
    dataset = CardDataset.from_labels_dict(merged_labels)

    train_loader, val_loader = get_train_val_loaders(
        dataset,
        batch_size=batch_size,
        val_split=val_split,
        train_transform=Config.TRANSFORMS['train'],
        val_transform=Config.TRANSFORMS['test'],
        shuffle=True
    )
    return train_loader, val_loader

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
    running_loss = 0.0

    # Calculate training loss
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad() # Reset gradients

        _, loss = forward_pass(model, images, labels, criterion, scaler)
        backward_step(loss, optimizer, scaler)

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)



def validate(model, loader, criterion, device):
    # Set model to evaluation mode
    model.eval()
    val_loss = 0.0
    # Calculate validation loss
    with torch.no_grad(): # Disable gradient computation for validation
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            _, loss = forward_pass(model, images, labels, criterion)
            val_loss += loss.item() * images.size(0)
    return val_loss / len(loader.dataset)

# -----------------------------
# Logging
# -----------------------------
def log_epoch_stats(epoch, train_loss, val_loss, optimizer, writer=None):
    """
    Logs training and validation losses for the given epoch.
    Optionally logs to TensorBoard if writer is provided.
    """
    lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.6f}")

    # Optionally log to TensorBoard
    if writer:
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Learning_Rate", lr, epoch)

# -----------------------------
# Checkpointing
# -----------------------------
def handle_checkpoints(model, val_loss, best_val_loss, epoch, checkpoint_dir, checkpoint_interval=5):
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, os.path.join(checkpoint_dir, 'best_model.pth'))

    # Periodic checkpoints
    if epoch % checkpoint_interval == 0:
        save_checkpoint(model, os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}.pth'))

    return best_val_loss

def save_checkpoint(model, filename):
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), filename)
    print(f"[INFO] Saved model: {filename}")


# -----------------------------
# Main Training Loop
# -----------------------------
def main():
    args = parse_args()
    num_classes = args.num_classes or Config.NUM_CLASSES

    # Load Data
    train_loader, val_loader = load_dataloaders(
        batch_size=args.batch_size,
        val_split=args.val_split,
        use_augmented=args.use_augmented
    )

    # Initialize all training objects in a single state
    state = prepare_training(num_classes=num_classes, log_dir=args.log_dir, lr=args.lr, patience=5)

    for epoch in range(1, args.epochs + 1):
        # Train for one epoch
        train_loss = train_one_epoch(state.model, train_loader, state.criterion, state.optimizer, state.device, state.scaler)
        
        # Validate
        val_loss = validate(state.model, val_loader, state.criterion, state.device)

        # Log stats
        log_epoch_stats(epoch, train_loss, val_loss, state.optimizer, state.writer)

        # Step scheduler (Adjusting the learning rate during training can help the model converge faster)
        state.scheduler.step()

        # Handle checkpoints (best model + periodic snapshots)
        state.best_val_loss = handle_checkpoints(
            state.model,
            val_loss,
            state.best_val_loss,
            epoch,
            checkpoint_dir=MODELS_DIR,
            checkpoint_interval=args.checkpoint_interval
        )

        # Early stopping
        state.early_stopping.step(val_loss)
        if state.early_stopping.early_stop:
            print("[INFO] Early stopping triggered")
            break

    state.writer.close()


if __name__ == "__main__":
    main()
