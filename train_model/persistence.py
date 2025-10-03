#persistence.py
import torch
import os

def save_checkpoint(training_state, heads, path, epoch):
    checkpoint = {
        "epoch": epoch,
        "model": training_state.model.state_dict(),
        "optimizer": training_state.optimizer.state_dict(),
        "scheduler": training_state.scheduler.state_dict() if training_state.scheduler else None,
        "scaler": training_state.scaler.state_dict() if training_state.scaler else None,
        "head_states": {}
    }

    # Save class_names for each head
    if heads:
        for head in heads:
            checkpoint["head_states"][head.name] = {
                "class_names": head.class_names
            }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    print(f"✅ Saved checkpoint to {path}")
    

def handle_checkpoints(training_state, heads, val_loss, best_val_loss, epoch, checkpoint_dir, checkpoint_interval=5):
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(training_state, heads, os.path.join(checkpoint_dir, 'best_model.pth'), epoch)

    # Periodic checkpoints
    if epoch % checkpoint_interval == 0:
        save_checkpoint(training_state, heads, os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}.pth'), epoch)

    return best_val_loss

def load_checkpoint(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint {path} not found")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device)
    print(f"🔄 Loaded checkpoint from {path}")
    return checkpoint

def apply_checkpoint(checkpoint, model=None, optimizer=None, scheduler=None, scaler=None):
    """Apply checkpoint to provided objects. Returns start_epoch."""
    if model is not None:
        model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    print(f"🔄 Applied checkpoint, resuming at epoch {start_epoch}")
    return start_epoch