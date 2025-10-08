import pytest
import tempfile
import json
from torch import nn
from unittest.mock import patch, MagicMock
from train_model.task.heads import ClassificationHead, IdentificationHead, ModifierHead

# --------------------------
# Fixtures
# --------------------------
@pytest.fixture
def temp_labels_files():
    """Temporary labels files."""
    original = {"img1.png": {"name": "A"}, "img2.png": {"name": "B"}}
    augmented = {"img3.png": {"name": "C"}}
    with tempfile.TemporaryDirectory() as tmpdir:
        orig_path = f"{tmpdir}/original.json"
        aug_path = f"{tmpdir}/augmented.json"
        with open(orig_path, 'w') as f:
            json.dump(original, f)
        with open(aug_path, 'w') as f:
            json.dump(augmented, f)
        yield orig_path, aug_path

# --------------------------
# Tests
# --------------------------
def test_classificationhead_init_defaults():
    head = ClassificationHead("test_head")
    assert head.name == "test_head"
    assert isinstance(head.batch_size, int)
    assert isinstance(head.val_split, float)
    assert isinstance(head.use_weighted_sampler, bool)
    # train_transform and val_transform should be callables from Config.TRANSFORMS
    assert callable(head.train_transform)
    assert callable(head.val_transform)
    assert head.num_classes == 0

def test_subclass_defaults_override():
    id_head = IdentificationHead("identification")
    mod_head = ModifierHead("modifier")
    # Subclass defaults override parent DEFAULTS
    assert type(id_head.train_transform) != type(mod_head.train_transform) or id_head.val_split != mod_head.val_split

@patch("train_model.heads.CardDataset.from_labels_dict")
@patch("train_model.heads.get_train_val_loaders")
def test_load_dataloaders(mock_get_loaders, mock_dataset):
    dummy_dataset = MagicMock()
    mock_dataset.side_effect = [dummy_dataset, dummy_dataset]
    dummy_train_loader = dummy_val_loader = "loader"
    mock_get_loaders.return_value = (dummy_train_loader, dummy_val_loader)

    head = IdentificationHead("identification")
    train_loader, val_loader = head.load_dataloaders(checkpoint=None)

    assert train_loader == dummy_train_loader
    assert val_loader == dummy_val_loader
    assert head.train_loader == dummy_train_loader
    assert head.val_loader == dummy_val_loader
    # Ensure class_names and num_classes are filled from dataset
    assert head.class_names == dummy_dataset.class_names
    assert head.num_classes == dummy_dataset.num_classes

def test_load_json_reads_files(temp_labels_files):
    orig_path, aug_path = temp_labels_files
    head = ClassificationHead("test_head", config={"labels_file": orig_path, "augmented_labels_file": aug_path})
    labels, aug_labels = head.load_json()
    assert "img1.png" in labels
    assert "img3.png" in aug_labels

@patch("train_model.heads.stratified_split_with_aug")
def test_split_labels_called(mock_split):
    mock_split.return_value = ({"train": "t"}, {"val": "v"})
    head = ClassificationHead("test_head")
    train_labels, val_labels = head.split_labels({}, {}, 0.1, "name")
    mock_split.assert_called_once()
    assert train_labels == {"train": "t"}
    assert val_labels == {"val": "v"}

@patch("train_model.heads.CardDataset.from_labels_dict")
@patch("train_model.heads.get_train_val_loaders")
def test_build_datasets_sets_class_names_and_num_classes(mock_get_loaders, mock_dataset):
    dummy_train_dataset = MagicMock()
    dummy_val_dataset = MagicMock()
    dummy_train_dataset.class_names = ["A", "B"]
    dummy_train_dataset.num_classes = 2
    dummy_val_dataset.class_names = ["A", "B"]
    dummy_val_dataset.num_classes = 2
    mock_dataset.side_effect = [dummy_train_dataset, dummy_val_dataset]
    mock_get_loaders.return_value = ("train_loader", "val_loader")

    head = ClassificationHead("test_head")
    train_dataset, val_dataset = head.build_datasets(lambda x: x, lambda x: x, "name", None)

    assert train_dataset == dummy_train_dataset
    assert val_dataset == dummy_val_dataset
    assert head.class_names == ["A", "B"]
    assert head.num_classes == 2

@patch("train_model.heads.get_train_val_loaders")
def test_get_dataloaders_pass_args(mock_get_loaders):
    mock_get_loaders.return_value = ("train_loader", "val_loader")
    head = ClassificationHead("test_head")
    dummy_train_dataset = MagicMock()
    dummy_val_dataset = MagicMock()
    train_loader, val_loader = head.get_dataloaders(dummy_train_dataset, dummy_val_dataset, 16, True)
    mock_get_loaders.assert_called_once_with(dummy_train_dataset, dummy_val_dataset, 16, True)
    assert train_loader == "train_loader"
    assert val_loader == "val_loader"

def test_build_criterion_returns_cross_entropy():
    head = ClassificationHead("test_head")
    criterion = head.build_criterion()
    assert isinstance(criterion, nn.CrossEntropyLoss)

def test_subclass_defaults_minimal_config():
    """Ensure subclasses inherit and override DEFAULTS correctly."""
    id_head = IdentificationHead("identification")
    mod_head = ModifierHead("modifier")
    # IdentificationHead overrides train_transform
    assert id_head.train_transform is not None
    # ModifierHead overrides field
    assert mod_head.field == "modifier"