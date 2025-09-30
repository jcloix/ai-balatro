import torch
from train_model.train_config import Config
from train_model.data_loader_utils import load_dataloaders 
from config.config import MODELS_DIR
import argparse
from train_model.train_loops import train_one_epoch, validate
from train_model.models import prepare_training
from train_model.logging import log_epoch_stats
from train_model.model_saving import handle_checkpoints 
from train_model.metrics import Metrics


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
    parser.add_argument("--no-augmented", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze feature extractor layers")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint")
    parser.add_argument("--use-weighted-sampler", action="store_true")
    parser.add_argument("--train-transform", type=str, default="train")
    parser.add_argument("--val-transform", type=str, default="test")
    parser.add_argument("--subset-only", action="store_true", help="Use only original images that have augmentations")

    return parser.parse_args()

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
        no_augmented=args.no_augmented,
        use_weighted_sampler=args.use_weighted_sampler,
        train_transform_mode=args.train_transform,
        val_transform_mode=args.val_transform,
        subset_only=args.subset_only
    )

    # Initialize all training objects in a single state
    state = prepare_training(
        num_classes=num_classes, 
        log_dir=args.log_dir, 
        lr=args.lr, 
        patience=5, 
        dataset_size=len(train_loader.dataset)
    )

    for epoch in range(1, args.epochs + 1):
        # Train for one epoch
        train_loss = train_one_epoch(state.model, train_loader, state.criterion, state.optimizer, state.device, state.scaler)
        
        # Validate
        val_metrics = validate(state.model, val_loader, state.criterion, state.device, compute_metrics=True, num_classes=num_classes)

        # Agregate metrics
        epoch_metrics = Metrics.from_epoch(train_loss, val_metrics)

        # Log stats
        log_epoch_stats(epoch, state.optimizer, epoch_metrics, state.writer, train_loader.classes)

        # Step scheduler (Adjusting the learning rate during training can help the model converge faster)
        if isinstance(state.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            state.scheduler.step(epoch_metrics.val_loss)
        else:
            state.scheduler.step()

        # Handle checkpoints (best model + periodic snapshots)
        state.best_val_loss = handle_checkpoints(
            state.model,
            epoch_metrics.val_loss,
            state.best_val_loss,
            epoch,
            checkpoint_dir=MODELS_DIR,
            checkpoint_interval=args.checkpoint_interval
        )

        # Early stopping
        state.early_stopping.step(epoch_metrics.val_loss)
        if state.early_stopping.early_stop:
            print("[INFO] Early stopping triggered")
            break

    state.writer.close()


if __name__ == "__main__":
    main()
