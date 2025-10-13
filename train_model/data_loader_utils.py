#data_loader_utils.py
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import defaultdict
import random
import math
import json
from train_model.train_config import Config 
# --------------------------
#  Group filenames by key
# --------------------------
def group_by_card(labels_dict, key="name"):
    grouped = defaultdict(list)
    for fname, meta in labels_dict.items():
        grouped[meta[key]].append(fname)
    return grouped

# --------------------------
# Pick validation originals (always 1 per class)
# --------------------------
def pick_val_originals(orig_fnames, val_split=0.1):
    """
    Returns (train_files, val_files) by selecting a fraction of originals for validation.
    
    Note:
        - If the number of originals is less than what val_split would select,
          this function will always put at least 1 original in the validation set.
        - val_split is applied fractionally, but never less than 1 original.
    
    Args:
        orig_fnames (list): list of original filenames for a class
        val_split (float): fraction of originals to assign to validation
    
    Returns:
        train_files (list), val_files (list)
    """
    rng = random.Random()
    orig_fnames = list(orig_fnames)  # copy
    rng.shuffle(orig_fnames)

    n_val = max(1, math.ceil(len(orig_fnames) * val_split))
    val_files = orig_fnames[:n_val]
    train_files = orig_fnames[n_val:]
    return train_files, val_files

# --------------------------
# Assign augmentations (all to train)
# --------------------------
def assign_augmentations(aug_fnames, train_files):
    train_files.extend(aug_fnames)
    return train_files

# --------------------------
# Build labels dict
# --------------------------
def build_labels_dict(files, original_labels, augmented_labels=None):
    labels_dict = {}
    for f in files:
        if f in original_labels:
            labels_dict[f] = original_labels[f]
        elif augmented_labels and f in augmented_labels:
            labels_dict[f] = augmented_labels[f]
    return labels_dict

# --------------------------
# Main stratified split
# --------------------------
def stratified_split_with_aug(original_labels, augmented_labels=None, val_split=0.1, key="name"):
    train_labels = {}
    val_labels = {}
    split_info = {}  # Store info per card_name

    orig_by_name = group_by_card(original_labels, key)
    aug_by_name  = group_by_card(augmented_labels, key) if augmented_labels else defaultdict(list)

    for card_name in orig_by_name:
        orig_fnames = orig_by_name[card_name]
        aug_fnames = aug_by_name.get(card_name, [])

        # Step 2: pick 1 val original
        train_files, val_files = pick_val_originals(orig_fnames, val_split)

        # Step 3: all augmentations → train
        train_files = assign_augmentations(aug_fnames, train_files)

        # Step 4: build dicts
        train_labels.update(build_labels_dict(train_files, original_labels, augmented_labels))
        val_labels.update(build_labels_dict(val_files, original_labels, augmented_labels))

        # Save split info
        split_info[card_name] = {
            "train": train_files,
            "val": val_files
        }

    # Save to JSON
    with open(Config.OUTPUT_SPLIT_TRAIN_VAL, "w") as f:
        json.dump(split_info, f, indent=4)

    print(f"[INFO] Train/val split saved to {Config.OUTPUT_SPLIT_TRAIN_VAL}")
    return train_labels, val_labels

# --------------------------
# Create the train and validation data loaders
# --------------------------
def get_train_val_loaders(train_dataset, val_dataset, batch_size=32, use_weighted_sampler=False):
    # Weighted sampler for training
    if use_weighted_sampler:
        train_labels_tensor = torch.tensor([train_dataset.labels_dict[fname] for fname in train_dataset.filenames], dtype=torch.long)
        class_weights = 1.0 / torch.bincount(train_labels_tensor).float()
        sample_weights = class_weights[train_labels_tensor].clone()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Attach classes
    train_loader.classes = train_dataset.class_names
    val_loader.classes = val_dataset.class_names

    return train_loader, val_loader
