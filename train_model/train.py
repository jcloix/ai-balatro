#train.py
import torch
from train_model.train_config import Config
from config.config import MODELS_DIR
import argparse
from train_model.train_loops import train_one_epoch, validate
from train_model.train_setup import prepare_training
from train_model.train_state import EpochResult
from train_model.persistence import handle_checkpoints 
from train_model.task.factory import create_head
from train_model.persistence import load_checkpoint
from train_model.strategies.factory import StrategyFactory
from train_model.task.heads import IdentificationHead, ModifierHead #Needed for the factory
import warnings


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

def filter_warnings():
    warnings.filterwarnings(
        "ignore",
        message="The number of unique classes is greater than 50% of the number of samples"
    )

# -----------------------------
# Main Training Loop
# -----------------------------
def main():
    filter_warnings()
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
        epoch_res_list = []
        for head in heads:
            epoch_res = EpochResult(epoch=epoch)
            # Train for one epoch
            train_loss = train_one_epoch(head, state, epoch_res)
            
            # Validate
            val_loss = validate(head, state, epoch_res)

            # Agregate metrics
            epoch_res_list.append(epoch_res)
            head.compute_metrics(state, epoch_res)

            # Log stats
            head.log_metrics(state, epoch_res, state.writer)
        # Compute Loss on all heads (Max)
        epoch_val_loss = max(epoch_res.val_loss for epoch_res in epoch_res_list)
        
        # Apply the scheduler step.
        state.scheduler.step(epoch_val_loss)
        
        # Handle checkpoints (best model + periodic snapshots)
        state.best_val_loss = handle_checkpoints(state, heads, epoch_val_loss, epoch, checkpoint_interval=args.checkpoint_interval)

        # Early stopping
        state.early_stopping.step(epoch_val_loss)
        if state.early_stopping.early_stop:
            print("[INFO] Early stopping triggered")
            break

    state.writer.close()


if __name__ == "__main__":
    main()
