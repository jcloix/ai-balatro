# tests/test_dataset.py
import os
import json
import tempfile
import pytest
from PIL import Image
from torchvision import transforms
from train_model.dataset import load_merged_labels, CardDataset, get_train_val_loaders

# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def temp_labels_files():
    # Create temporary original labels
    original = {
        "img1.png": {"label": 0},
        "img2.png": {"label": 1}
    }
    augmented = {
        "img3.png": {"label": 2}
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        orig_path = os.path.join(tmpdir, "original.json")
        aug_path = os.path.join(tmpdir, "augmented.json")
        save_path = os.path.join(tmpdir, "merged.json")

        with open(orig_path, 'w') as f:
            json.dump(original, f)
        with open(aug_path, 'w') as f:
            json.dump(augmented, f)

        yield orig_path, aug_path, save_path, tmpdir

@pytest.fixture
def dummy_images(tmp_path):
    paths = []
    for i in range(3):
        img_path = tmp_path / f"img{i+1}.png"
        img = Image.new("RGB", (64, 64), color=(i*50, i*50, i*50))
        img.save(img_path)
        paths.append(str(img_path))
    return paths

# -----------------------------
# Tests
# -----------------------------
def test_load_merged_labels(temp_labels_files):
    orig_path, aug_path, save_path, _ = temp_labels_files

    merged = load_merged_labels(orig_path, aug_path, save_path)
    assert len(merged) == 3
    assert "img1.png" in merged and "img3.png" in merged

    # Check that file was saved
    assert os.path.exists(save_path)
    with open(save_path, 'r') as f:
        saved = json.load(f)
    assert saved == merged

def test_card_dataset_from_labels_dict(dummy_images):
    labels_dict = {path: {"label": idx} for idx, path in enumerate(dummy_images)}
    dataset = CardDataset.from_labels_dict(labels_dict)

    assert len(dataset) == len(dummy_images)
    img, label = dataset[0]
    assert isinstance(img, Image.Image)
    assert isinstance(label, int)
    assert label == 0

def test_get_train_val_loaders(dummy_images):
    labels_dict = {path: {"label": idx} for idx, path in enumerate(dummy_images)}
    dataset = CardDataset.from_labels_dict(labels_dict)

    # Minimal transform to convert PIL -> Tensor
    tensor_transform = transforms.ToTensor()

    train_loader, val_loader = get_train_val_loaders(
        dataset,
        batch_size=2,
        val_split=0.33,
        train_transform=tensor_transform,
        val_transform=tensor_transform
    )

    # Now batches will contain tensors, not PIL Images
    for batch_imgs, batch_labels in train_loader:
        assert batch_imgs[0].dtype  # check it’s a tensor
        assert all(isinstance(lbl.item(), int) for lbl in batch_labels)
