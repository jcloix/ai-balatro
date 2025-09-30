# train_model/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from PIL import Image
import json
import os

def load_merged_labels(original_labels_path, augmented_labels_path=None, save_path=None, subset_only=True):
    """
    Load original and augmented labels and merge them in memory.
    If save_path is provided, write the merged mapping to that file.
    """
    with open(original_labels_path, 'r') as f:
        labels = json.load(f)

    if augmented_labels_path and os.path.exists(augmented_labels_path):
        with open(augmented_labels_path, 'r') as f:
            aug_labels = json.load(f)

        if subset_only:
            # keep only original images that exist in augmented_labels
            labels = {k: v for k, v in labels.items() if k in aug_labels}

        # Merge augmented labels
        labels.update(aug_labels)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(labels, f, indent=2)

    return labels


class CardDataset(Dataset):
    """
    Dataset constructed from a labels dict (filename -> metadata).
    """
    def __init__(self, labels_dict, transform=None):
        self.data = labels_dict
        self.filenames = list(self.data.keys())
        self.transform = transform  # store transform

        # Build a sorted list of unique class names (human-readable)
        self.class_names = sorted({v['name'] for v in self.data.values()})
        self.classes = self.class_names   

        # Map class name → integer label
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}

        # Store integer labels for each file
        self.labels_dict = {fname: self.class_to_idx[self.data[fname]['name']] for fname in self.filenames}


    @classmethod
    def from_labels_dict(cls, labels_dict):
        return cls(labels_dict=labels_dict)

    @classmethod
    def from_label_paths(cls, original_labels_path, augmented_labels_path=None, save_path=None):
        labels = load_merged_labels(original_labels_path, augmented_labels_path, save_path=save_path)
        return cls.from_labels_dict(labels)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        label = self.labels_dict[fname]

        # Use full_path if available, fallback to fname
        meta = self.data[fname]
        img_path = meta.get("full_path", fname)

        image = Image.open(img_path).convert('RGB')

        if self.transform:  # apply transform if provided
            image = self.transform(image)

        return image, label


# -----------------------------
# Helper: Train/Val Split Loader
# -----------------------------
def get_train_val_loaders(dataset, batch_size=32, val_split=0.1, train_transform=None, val_transform=None, shuffle=True, use_weighted_sampler=False):
    """
    Split a dataset into train/validation loaders, optionally applying different transforms.
    """
    # Determine split sizes between train and val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Apply different transforms for train/val
    if train_transform:
        train_dataset.dataset.transform = train_transform
    if val_transform:
        val_dataset.dataset.transform = val_transform

    # For the training loader, if the dataset is imbalanced, use a weighted sampler (activable by argument).
    if use_weighted_sampler:
        # Get labels only for the subset
        train_labels = [dataset.labels_dict[dataset.filenames[i]] for i in train_dataset.indices]

        label_counts = torch.bincount(torch.tensor(train_labels, dtype=torch.long))
        class_weights = 1.0 / label_counts.float()
        sample_weights = torch.tensor([class_weights[label] for label in train_labels], dtype=torch.float)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Attach classes to loaders
    train_loader.classes = dataset.class_names
    val_loader.classes = dataset.class_names

    return train_loader, val_loader
