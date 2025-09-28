import os
import tempfile
import json
from pathlib import Path
import pytest
from unittest import mock

# Assuming the module is named save_augmented_labels.py
import augment_dataset.save_augmented_labels as sal

@pytest.fixture
def setup_labels_and_augmented_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Original labels file
        labels_file = os.path.join(tmpdir, "labels.json")
        original_labels = {
            "cluster1_card1_id1.png": {
                "name": "Dna",
                "type": "Joker",
                "rarity": "Rare",
                "modifier": "Base",
                "cluster": 1,
                "card_group": 1
            }
        }
        with open(labels_file, "w", encoding="utf-8") as f:
            json.dump(original_labels, f)

        # Augmented images
        aug_dir = os.path.join(tmpdir, "augmented")
        os.makedirs(aug_dir, exist_ok=True)
        aug_filenames = ["cluster1_card1_id1_aug0.png", "cluster1_card1_id1_aug1.png"]
        for fname in aug_filenames:
            Path(os.path.join(aug_dir, fname)).touch()

        yield {
            "tmpdir": tmpdir,
            "labels_file": labels_file,
            "aug_dir": aug_dir,
            "original_labels": original_labels,
            "aug_filenames": aug_filenames
        }

def test_find_parent_and_build_augmented_labels(setup_labels_and_augmented_dir, monkeypatch):
    data = setup_labels_and_augmented_dir

    # Patch config paths in the module
    monkeypatch.setattr(sal, "DATASET_AUGMENTED_DIR", data["aug_dir"])
    monkeypatch.setattr(sal, "LABELS_FILE", data["labels_file"])

    # Test find_parent
    parent, aug_type = sal.find_parent("cluster1_card1_id1_aug0.png")
    assert parent == "cluster1_card1_id1.png"
    assert aug_type == "aug0"

    parent, aug_type = sal.find_parent("no_aug_here.png")
    assert parent is None and aug_type is None

    # Test build_augmented_labels
    augmented_labels = sal.build_augmented_labels(data["original_labels"])
    assert len(augmented_labels) == len(data["aug_filenames"])
    for fname, meta in augmented_labels.items():
        assert "parent" in meta and "augmentation" in meta
        assert meta["parent"] == "cluster1_card1_id1.png"

def test_main_creates_json_file(setup_labels_and_augmented_dir, monkeypatch):
    data = setup_labels_and_augmented_dir

    monkeypatch.setattr(sal, "DATASET_AUGMENTED_DIR", data["aug_dir"])
    monkeypatch.setattr(sal, "LABELS_FILE", data["labels_file"])
    
    # Save to a temp file instead of default
    temp_aug_file = os.path.join(data["tmpdir"], "augmented.json")
    monkeypatch.setattr(sal, "AUGMENTED_LABELS_FILE", temp_aug_file)

    sal.main()

    # Verify JSON file created correctly
    assert os.path.exists(temp_aug_file)
    with open(temp_aug_file, "r", encoding="utf-8") as f:
        aug_labels = json.load(f)
    assert len(aug_labels) == len(data["aug_filenames"])
    for fname, meta in aug_labels.items():
        assert meta["parent"] == "cluster1_card1_id1.png"
