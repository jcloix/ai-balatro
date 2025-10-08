#persistence.py
import torch
import os
from config.config import MODELS_DIR 

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
    

def handle_checkpoints(training_state, heads, val_loss, epoch, checkpoint_dir=MODELS_DIR, checkpoint_interval=5):
    best_val_loss = training_state.best_val_loss
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
        safe_load_optimizer_state(optimizer, checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
        safe_load_scheduler_state(scheduler, checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    print(f"🔄 Applied checkpoint, resuming at epoch {start_epoch}")
    return start_epoch

def safe_load_optimizer_state(optimizer, checkpoint_optimizer_state):
    """
    Load optimizer state from checkpoint safely, ignoring missing params.
    """
    new_state_dict = optimizer.state_dict()
    old_state_dict = checkpoint_optimizer_state

    # Copy param groups
    for new_group, old_group in zip(new_state_dict['param_groups'], old_state_dict['param_groups']):
        # Only copy hyperparameters that exist
        for key in ['lr', 'momentum', 'betas', 'weight_decay']:
            if key in old_group:
                new_group[key] = old_group[key]

    # Copy state for matching params
    new_state = new_state_dict['state']
    old_state = old_state_dict['state']

    for param_id, state in new_state.items():
        if param_id in old_state:
            # copy only the keys that exist (like momentum_buffer, exp_avg, etc.)
            for k, v in old_state[param_id].items():
                state[k] = v

    optimizer.load_state_dict(new_state_dict)
    return optimizer


def safe_load_scheduler_state(scheduler_wrapper, checkpoint_scheduler_state):
    """
    scheduler_wrapper: your SchedulerStrategy instance
    checkpoint_scheduler_state: dict from state_dict()
    """
    # The scheduler is a custom scheduler wrapping a real one
    inner_scheduler = getattr(scheduler_wrapper, "scheduler", None)
    
    if inner_scheduler and type(inner_scheduler) == type(checkpoint_scheduler_state):
        try:
            inner_scheduler.load_state_dict(checkpoint_scheduler_state)
        except Exception as e:
            print(f"⚠️ Scheduler state could not be fully loaded: {e}")
    else:
        # fallback: just set last_epoch if present
        if inner_scheduler and 'last_epoch' in checkpoint_scheduler_state:
            inner_scheduler.last_epoch = checkpoint_scheduler_state['last_epoch']
        print("⚠️ Scheduler type changed, state partially applied or ignored")

    return scheduler_wrapper
