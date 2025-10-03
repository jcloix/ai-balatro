#train.py
import torch
from train_model.train_config import Config
from config.config import MODELS_DIR
import argparse
from train_model.train_loops import train_one_epoch, validate
from train_model.train_setup import prepare_training
from train_model.logging import log_epoch_stats
from train_model.persistence import handle_checkpoints 
from train_model.metrics import Metrics
from train_model.factory import create_head
from train_model.heads import IdentificationHead, ModifierHead 
from train_model.persistence import load_checkpoint


# -----------------------------
# Argument Parser
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train Balatro card recognition model")
    parser.add_argument(
        "--tasks",
        nargs="*",
        choices=["identification", "modifier"],
        default=["identification", "modifier"],
        help="List of tasks (choose from identification, modifier, e.g. train.py --tasks identification modifier)"
    )
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS)
    parser.add_argument("--lr", type=float, help="Override learning rate for all heads")
    parser.add_argument("--patience", type=int, default=Config.PATIENCE)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--checkpoint-interval", type=int, default=Config.CHECKPOINT_INTERVAL)
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze feature extractor layers")
    parser.add_argument("--resume",type=str,default=None,help="Resume training from checkpoint(s)")
    parser.add_argument("--use-weighted-sampler", action="store_true")

    return parser.parse_args()

# -----------------------------
# Main Training Loop
# -----------------------------
def main():
    args = parse_args()

    # Load checkpoint first
    checkpoint = load_checkpoint(args.resume) if args.resume else None

    # Create the correct HEAD(s) strategy from the factory
    heads = [create_head(name, vars(args)) for name in args.tasks]

    # Load Data
    for head in heads:
        head.load_dataloaders(checkpoint)

    # Prepare the model and training config
    state = prepare_training(heads, args.log_dir, args.lr, args.patience, args.freeze_backbone, checkpoint)

    for epoch in range(state.start_epoch, args.epochs + 1):
        metrics = []
        for head in heads:
            # Train for one epoch
            train_loss = train_one_epoch(state.model, head.train_loader, head.criterion, state.optimizer, state.device, state.scaler, head.name)
            
            # Validate
            val_metrics = validate(state.model, head.val_loader, head.criterion, state.device, compute_metrics=True, task_name=head.name, num_classes=head.num_classes)

            # Agregate metrics
            epoch_metrics = Metrics.from_epoch(train_loss, val_metrics)
            metrics.append(epoch_metrics)

            # Log stats
            log_epoch_stats(epoch, state.optimizer, epoch_metrics, state.writer, head.train_loader.classes, head.name)
        # Compute Loss on all heads (Average)
        epoch_val_loss = sum(metric.val_loss for metric in metrics) / len(metrics)
        # Step scheduler (Adjusting the learning rate during training can help the model converge faster)
        if isinstance(state.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            state.scheduler.step(epoch_val_loss)
        else:
            state.scheduler.step()
        
        # Handle checkpoints (best model + periodic snapshots)
        state.best_val_loss = handle_checkpoints(
            state,
            heads,
            epoch_val_loss,
            state.best_val_loss,
            epoch,
            checkpoint_dir=MODELS_DIR,
            checkpoint_interval=args.checkpoint_interval,
        )

        # Early stopping
        state.early_stopping.step(epoch_val_loss)
        if state.early_stopping.early_stop:
            print("[INFO] Early stopping triggered")
            break

    state.writer.close()


if __name__ == "__main__":
    main()
