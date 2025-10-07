#train.py
import torch
from train_model.train_config import Config
from config.config import MODELS_DIR
import argparse
from train_model.train_loops import train_one_epoch, validate
from train_model.train_setup import prepare_training, build_model
from train_model.logging import log_epoch_stats
from train_model.persistence import handle_checkpoints 
from train_model.metrics import Metrics
from train_model.factory import create_head
from train_model.heads import IdentificationHead, ModifierHead 
from train_model.persistence import load_checkpoint
from train_model.strategies.factory import StrategyFactory


# -----------------------------
# Argument Parser
# -----------------------------
# -----------------------------
# Argument Parser
# -----------------------------
# -----------------------------
# Argument Parser
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train Balatro card recognition model")

    # ---- General training args ----
    parser.add_argument(
        "--tasks",
        nargs="*",
        choices=["identification", "modifier"],
        default=["identification", "modifier"],
        help="List of tasks (choose from identification, modifier, e.g. train.py --tasks identification modifier)"
    )
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS)
    parser.add_argument("--patience", type=int, default=Config.PATIENCE)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--checkpoint-interval", type=int, default=Config.CHECKPOINT_INTERVAL)
    parser.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint(s)")
    parser.add_argument("--use-weighted-sampler", action="store_true")

    # ---- Strategy selection ----
    parser.add_argument("--freeze-strategy", type=str, default="none",
                        choices=["none", "high", "mid", "all"], help="Choose backbone freezing strategy")
    parser.add_argument("--optimizer", type=str, default="simple",
                        choices=["simple", "group"], help="Choose optimizer strategy")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "plateau", "step", "none"], help="Choose scheduler strategy")

    # ---- Freeze strategy params ----
    parser.add_argument("--freeze-layers-high", nargs="*", default=["layer3", "layer4"],
                        help="Layers to unfreeze in 'high' freeze strategy")
    parser.add_argument("--freeze-layers-mid", nargs="*", default=["layer2", "layer3", "layer4"],
                        help="Layers to unfreeze in 'mid' freeze strategy")

    # ---- Optimizer params ----
    parser.add_argument("--optimizer-lr", type=float, default=1e-4,
                        help="Learning rate for SimpleAdam")
    parser.add_argument("--optimizer-lr-backbone", type=float, default=1e-5,
                        help="Learning rate for backbone in GroupAdamW")
    parser.add_argument("--optimizer-lr-heads", type=float, default=1e-4,
                        help="Learning rate for heads in GroupAdamW")
    parser.add_argument("--optimizer-weight-decay", type=float, default=1e-4,
                        help="Weight decay for GroupAdamW")

    # ---- Scheduler params ----
    parser.add_argument("--scheduler-tmax", type=int, default=50,
                        help="T_max for CosineAnnealingLR")
    parser.add_argument("--scheduler-factor", type=float, default=0.1,
                        help="Factor for ReduceLROnPlateau")
    parser.add_argument("--scheduler-patience", type=int, default=5,
                        help="Patience for ReduceLROnPlateau")
    parser.add_argument("--scheduler-step-size", type=int, default=5,
                        help="Step size for StepLR")
    parser.add_argument("--scheduler-gamma", type=float, default=0.5,
                        help="Gamma for StepLR")

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
    strategy = StrategyFactory.from_cli(args)
    state = prepare_training(heads, strategy, args.log_dir, args.patience, checkpoint)

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
        epoch_val_loss = max(metric.val_loss for metric in metrics)
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
