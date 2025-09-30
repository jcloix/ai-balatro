import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from train_model.dataset import CardDataset, get_train_val_loaders, load_merged_labels
from train_model.train_config import Config
from config.config import LABELS_FILE, AUGMENTED_LABELS_FILE, MERGED_LABELS_FILE

# -----------------------------
# DataLoader Setup
# -----------------------------
def load_dataloaders(
        batch_size, 
        val_split, 
        no_augmented=False, 
        use_weighted_sampler=False,
        train_transform_mode="train",
        val_transform_mode="test",
        subset_only=False
    ):

    # Merge labels once and save snapshot
    merged_labels = load_merged_labels(LABELS_FILE, None if no_augmented else AUGMENTED_LABELS_FILE, MERGED_LABELS_FILE, subset_only)

    # Create dataset (no internal transforms)
    dataset = CardDataset.from_labels_dict(merged_labels)

    # Get train/val loaders with appropriate transforms
    return get_train_val_loaders(
        dataset,
        batch_size=batch_size,
        val_split=val_split,
        train_transform=Config.TRANSFORMS[train_transform_mode],
        val_transform=Config.TRANSFORMS[val_transform_mode],
        shuffle=True,
        use_weighted_sampler=use_weighted_sampler
    )
