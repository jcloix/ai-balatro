import os
import tempfile
import json
import sys
from pathlib import Path
import pytest

import augment_dataset.save_augmented_labels as sal

@pytest.fixture
def setup_labels_and_aug_dir():
    """Setup a temporary labels file and augmented directory with images"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Original labels
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
        aug_files = ["cluster1_card1_id1_aug0.png", "cluster1_card1_id1_aug1.png"]
        for fname in aug_files:
            Path(os.path.join(aug_dir, fname)).touch()

        yield {
            "tmpdir": tmpdir,
            "labels_file": labels_file,
            "aug_dir": aug_dir,
            "original_labels": original_labels,
            "aug_files": aug_files
        }

def test_find_parent():
    # Normal case
    parent, aug = sal.find_parent("cluster5_card17_id302_aug3.png")
    assert parent == "cluster5_card17_id302.png"
    assert aug == "aug3"

    # No augmentation pattern
    parent, aug = sal.find_parent("no_aug.png")
    assert parent is None and aug is None

def test_build_augmented_labels(setup_labels_and_aug_dir):
    data = setup_labels_and_aug_dir
    aug_labels = sal.build_augmented_labels(data["aug_dir"], data["original_labels"])
    
    assert len(aug_labels) == len(data["aug_files"])
    for fname, meta in aug_labels.items():
        assert "parent" in meta
        assert "augmentation" in meta
        assert meta["parent"] == "cluster1_card1_id1.png"
        assert meta["full_path"].endswith(fname)

def test_main_creates_json_file(setup_labels_and_aug_dir, monkeypatch):
    data = setup_labels_and_aug_dir
    output_file = os.path.join(data["aug_dir"], "augmented.json")

    # Patch sys.argv so argparse reads our test paths
    monkeypatch.setattr(sys, "argv", [
        "prog",
        "--aug-dir", data["aug_dir"],
        "--labels", data["labels_file"]
    ])

    sal.main()

    # Check file exists and contains correct entries
    assert os.path.exists(output_file)
    with open(output_file, "r", encoding="utf-8") as f:
        aug_labels = json.load(f)
    
    assert len(aug_labels) == len(data["aug_files"])
    for fname, meta in aug_labels.items():
        assert meta["parent"] == "cluster1_card1_id1.png"
