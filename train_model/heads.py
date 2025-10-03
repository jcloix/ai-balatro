#head.py
import json
import os
from abc import ABC
from train_model.factory import register_head
from train_model.train_config import Config
from config.config import LABELS_FILE, AUGMENTED_LABELS_FILE
from train_model.data_loader_utils import stratified_split_with_aug, get_train_val_loaders
from train_model.dataset import CardDataset
from torch import nn

class ClassificationHead(ABC):
    DEFAULTS = {
        "batch_size":Config.BATCH_SIZE,
        "val_split":Config.VAL_SPLIT,
        "use_weighted_sampler":True,
        "train_transform":"train",
        "val_transform":"test",
        "labels_file":LABELS_FILE,
        "augmented_labels_file":AUGMENTED_LABELS_FILE,
        "field":"name"
    }
    def __init__(self, name, config=None):
        self.name = name

        cfg = dict(ClassificationHead.DEFAULTS)
        cfg.update(getattr(self, "DEFAULTS", {}))  # subclass defaults
        if config:
            cfg.update(config)  # runtime overrides

        # Store values
        self.batch_size = cfg["batch_size"]
        self.val_split = cfg["val_split"]
        self.use_weighted_sampler = cfg["use_weighted_sampler"]
        self.train_transform = Config.TRANSFORMS[cfg["train_transform"]]
        self.val_transform = Config.TRANSFORMS[cfg["val_transform"]]
        self.labels_file = cfg["labels_file"]
        self.augmented_labels_file = cfg["augmented_labels_file"]
        self.field = cfg["field"]
        self.num_classes=0
        self.criterion = self.build_criterion()

    def load_dataloaders(self, checkpoint):
        """
        Load datasets and dataloaders.
        If checkpoint is provided, use saved class_names to keep label alignment.
        """
        head_state = None
        if checkpoint and "head_states" in checkpoint:
            head_state = checkpoint["head_states"].get(self.name)

        class_names = head_state.get("class_names") if head_state else None
        # Build the datasets
        train_dataset, val_dataset = self.build_datasets(self.train_transform, self.val_transform, self.field, class_names)
        # Build the dataloaders
        self.train_loader, self.val_loader = self.get_dataloaders(train_dataset, val_dataset, self.batch_size, self.use_weighted_sampler)
        return self.train_loader, self.val_loader

    def load_json(self):
        aug_labels={}
        with open(self.labels_file, 'r') as f:
            labels = json.load(f)
        if self.augmented_labels_file and os.path.exists(self.augmented_labels_file):
            with open(self.augmented_labels_file, 'r') as f:
                aug_labels = json.load(f)
        return labels, aug_labels
    
    def split_labels(self, labels, aug_labels, val_split, key):
        return stratified_split_with_aug(labels, aug_labels, val_split, key)
    
    def build_datasets(self, train_transform, val_transform, field, class_names):
         # load the JSON files
        labels, aug_labels = self.load_json()
        # Split the labels in train and val
        train_labels, val_labels = self.split_labels(labels, aug_labels, self.val_split, self.field)
        # Create datasets
        train_dataset, val_dataset = CardDataset.from_labels_dict(train_labels, field, train_transform, class_names), CardDataset.from_labels_dict(val_labels, field, val_transform, class_names)
        # Fill the class_names and num_classes correctly since we loaded data into our head
        self.class_names  = train_dataset.class_names 
        self.num_classes = train_dataset.num_classes
        return train_dataset, val_dataset
    
    def get_dataloaders(self, train_dataset, val_dataset, batch_size, use_weighted_sampler):
        return get_train_val_loaders(train_dataset, val_dataset, batch_size, use_weighted_sampler)
    
    def build_criterion(self):
        return nn.CrossEntropyLoss()


@register_head("identification")
class IdentificationHead(ClassificationHead):
    DEFAULTS = {
        "train_transform":"heavy",
        "val_transform":"test",
        "labels_file":LABELS_FILE,
        "augmented_labels_file":AUGMENTED_LABELS_FILE
    }


@register_head("modifier")
class ModifierHead(ClassificationHead):
    DEFAULTS = {
        "val_split":0.1,
        "train_transform":"light",
        "val_transform":"test",
        "labels_file":LABELS_FILE,
        "augmented_labels_file":"data/modifier_augmented/augmented.json",
        "field":"modifier"
    }
