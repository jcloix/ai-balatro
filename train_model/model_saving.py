#model_saving.py
import torch
import os
from config.config import  MODELS_DIR

def save_checkpoint(model, filename):
    checkpoint_dir = os.path.dirname(filename)  # derive dir from filename
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), filename)
    print(f"[INFO] Saved model: {filename}")

def handle_checkpoints(model, val_loss, best_val_loss, epoch, checkpoint_dir, checkpoint_interval=5):
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, os.path.join(checkpoint_dir, 'best_model.pth'))

    # Periodic checkpoints
    if epoch % checkpoint_interval == 0:
        save_checkpoint(model, os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}.pth'))

    return best_val_loss
