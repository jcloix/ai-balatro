# test_data_loader_utils.py
import pytest
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from train_model import data_loader_utils

# -------------------------
# Fixtures
# -------------------------
@pytest.fixture
def dummy_labels():
    return {
        "img1.png": {"name": "Baron"},
        "img2.png": {"name": "Baron"},
        "img3.png": {"name": "Joker"},
        "img4.png": {"name": "Joker"},
        "img5.png": {"name": "Mime"}
    }

@pytest.fixture
def dummy_augmented_labels():
    return {
        "img1_aug.png": {"name": "Baron"},
        "img3_aug.png": {"name": "Joker"},
    }

# -------------------------
# group_by_card tests
# -------------------------
def test_group_by_card(dummy_labels):
    grouped = data_loader_utils.group_by_card(dummy_labels, key="name")
    assert set(grouped.keys()) == {"Baron", "Joker", "Mime"}
    assert set(grouped["Baron"]) == {"img1.png", "img2.png"}
    assert set(grouped["Mime"]) == {"img5.png"}

# -------------------------
# pick_val_originals tests
# -------------------------
def test_pick_val_originals_min_one():
    # If val_split would be <1, we still get 1 in val
    train, val = data_loader_utils.pick_val_originals(["a.png"], val_split=0.1)
    assert len(val) == 1
    assert len(train) == 0

def test_pick_val_originals_fraction():
    files = ["a.png", "b.png", "c.png", "d.png", "e.png"]
    train, val = data_loader_utils.pick_val_originals(files, val_split=0.4)
    # 5 * 0.4 = 2, ceil -> 2 val files
    assert len(val) == 2
    assert len(train) == 3
    assert set(train + val) == set(files)

# -------------------------
# assign_augmentations tests
# -------------------------
def test_assign_augmentations_adds_to_train():
    train_files = ["a.png", "b.png"]
    aug_files = ["a_aug.png", "b_aug.png"]
    updated_train = data_loader_utils.assign_augmentations(aug_files, train_files.copy())
    assert set(updated_train) == {"a.png", "b.png", "a_aug.png", "b_aug.png"}

# -------------------------
# build_labels_dict tests
# -------------------------
def test_build_labels_dict_merges_labels(dummy_labels, dummy_augmented_labels):
    files = ["img1.png", "img3_aug.png"]
    labels_dict = data_loader_utils.build_labels_dict(files, dummy_labels, dummy_augmented_labels)
    assert labels_dict["img1.png"]["name"] == "Baron"
    assert labels_dict["img3_aug.png"]["name"] == "Joker"

# -------------------------
# stratified_split_with_aug tests
# -------------------------
def test_stratified_split_with_aug_split(dummy_labels, dummy_augmented_labels):
    train, val = data_loader_utils.stratified_split_with_aug(
        dummy_labels, dummy_augmented_labels, val_split=0.5
    )
    # Originals must be split, augmented only in train
    assert "img1_aug.png" in train
    assert "img1_aug.png" not in val
    # Each class must have at least one in val
    class_in_val = {v["name"] for v in val.values()}
    assert class_in_val.issubset({"Baron", "Joker", "Mime"})

# -------------------------
# Dummy Dataset for DataLoader tests
# -------------------------
class DummyDataset(Dataset):
    def __init__(self):
        self.filenames = ["f1", "f2", "f3"]
        self.labels_dict = {"f1": 0, "f2": 1, "f3": 0}
        self.class_names = ["A", "B"]
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        return idx, self.labels_dict[self.filenames[idx]]

# -------------------------
# get_train_val_loaders tests
# -------------------------
def test_get_train_val_loaders_no_sampler():
    train_dataset = DummyDataset()
    val_dataset = DummyDataset()
    train_loader, val_loader = data_loader_utils.get_train_val_loaders(train_dataset, val_dataset, batch_size=2)
    # DataLoader should have correct length
    assert len(train_loader) == 2  # batch size
    assert len(val_loader) == 2
    # Classes attached
    assert train_loader.classes == ["A", "B"]
    assert val_loader.classes == ["A", "B"]

def test_get_train_val_loaders_weighted_sampler():
    train_dataset = DummyDataset()
    val_dataset = DummyDataset()
    train_loader, val_loader = data_loader_utils.get_train_val_loaders(train_dataset, val_dataset, batch_size=2, use_weighted_sampler=True)
    # Sampler exists
    assert hasattr(train_loader, "sampler")
    # Classes attached
    assert train_loader.classes == ["A", "B"]
    assert val_loader.classes == ["A", "B"]
