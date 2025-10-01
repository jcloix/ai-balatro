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
    """Create temporary original and augmented labels using 'name' field."""
    original = {"img1.png": {"name": "Joker"}, "img2.png": {"name": "Mime"}}
    augmented = {"img3.png": {"name": "Baron"}}

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
    """Create dummy PNG images."""
    paths = []
    for i in range(10):  # 10 samples instead of 3
        img_path = tmp_path / f"img{i+1}.png"
        img = Image.new("RGB", (64, 64), color=(i * 20, i * 20, i * 20))
        img.save(img_path)
        paths.append(str(img_path))
    return paths


# -----------------------------
# Tests
# -----------------------------
def test_load_merged_labels(temp_labels_files):
    orig_path, aug_path, save_path, _ = temp_labels_files

    # Merge original + augmented
    merged = load_merged_labels(orig_path, aug_path, save_path, False)
    assert len(merged) == 3
    assert "img1.png" in merged and "img3.png" in merged

    # File was saved correctly
    assert os.path.exists(save_path)
    with open(save_path, 'r') as f:
        saved = json.load(f)
    assert saved == merged

    # Test without augmented labels
    merged2 = load_merged_labels(orig_path)
    assert len(merged2) == 2
    assert "img1.png" in merged2 and "img2.png" in merged2


def test_card_dataset_from_labels_dict(dummy_images):
    labels_dict = {path: {"name": f"class_{idx}"} for idx, path in enumerate(dummy_images)}
    dataset = CardDataset.from_labels_dict(labels_dict)

    # Length check
    assert len(dataset) == len(dummy_images)

    # __getitem__ returns PIL image and integer label
    img, label = dataset[0]
    assert isinstance(img, Image.Image)
    assert isinstance(label, int)
    assert 0 <= label < len(dummy_images)

    # class_names are sorted and unique
    expected_classes = sorted([f"class_{i}" for i in range(len(dummy_images))])
    assert dataset.class_names == expected_classes

    # Test transform application
    tensor_transform = transforms.ToTensor()
    dataset.transform = tensor_transform
    img, _ = dataset[0]
    import torch
    assert isinstance(img, torch.Tensor)

    # Index out of range raises IndexError
    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]


def test_card_dataset_from_label_paths(temp_labels_files):
    orig_path, aug_path, save_path, _ = temp_labels_files
    dataset = CardDataset.from_label_paths(orig_path, aug_path, save_path)
    assert len(dataset) == 1

    # class_names are strings
    for cls in dataset.class_names:
        assert isinstance(cls, str)


def test_get_train_val_loaders(dummy_images):
    labels_dict = {path: {"name": f"class_{idx}"} for idx, path in enumerate(dummy_images)}
    dataset = CardDataset.from_labels_dict(labels_dict)

    tensor_transform = transforms.ToTensor()
    dataset.transform = tensor_transform

    train_loader, val_loader = get_train_val_loaders(
        dataset,
        batch_size=2,
        val_split=0.33,
        train_transform=tensor_transform,
        val_transform=tensor_transform
    )

    # Total number of samples
    total_samples = sum(len(batch[0]) for batch in train_loader) + sum(len(batch[0]) for batch in val_loader)
    assert total_samples == len(dataset)

    # Batches are tensors and labels are integers
    for batch_imgs, batch_labels in train_loader:
        import torch
        assert isinstance(batch_imgs[0], torch.Tensor)
        assert all(isinstance(lbl.item(), int) for lbl in batch_labels)

    # Validate val loader size roughly matches split
    val_len = len(val_loader.dataset)
    assert 0 < val_len < len(dataset)


def test_weighted_sampler(dummy_images):
    """Ensure WeightedRandomSampler can be used without error."""
    # Imbalanced dataset: 2 "Mars", 1 "Venus"
    labels_dict = {
        dummy_images[0]: {"name": "Mars"},
        dummy_images[1]: {"name": "Mars"},
        dummy_images[2]: {"name": "Venus"}
    }
    dataset = CardDataset.from_labels_dict(labels_dict)

    # Apply transform to convert PIL -> tensor
    tensor_transform = transforms.ToTensor()
    dataset.transform = tensor_transform

    train_loader, _ = get_train_val_loaders(
        dataset,
        batch_size=2,
        val_split=0.33,
        use_weighted_sampler=True
    )

    # Ensure loader is iterable and labels are integers
    for batch_imgs, batch_labels in train_loader:
        import torch
        assert len(batch_imgs) > 0
        assert isinstance(batch_imgs[0], torch.Tensor)
        assert all(isinstance(lbl.item(), int) for lbl in batch_labels)

def test_labels_match_class_to_idx(dummy_images):
    """Check that integer labels match the class_to_idx mapping."""
    labels_dict = {
        dummy_images[0]: {"name": "Mars"},
        dummy_images[1]: {"name": "Venus"},
        dummy_images[2]: {"name": "Jupiter"}
    }
    dataset = CardDataset.from_labels_dict(labels_dict)

    # Apply tensor transform to avoid collate errors if needed
    from torchvision import transforms
    dataset.transform = transforms.ToTensor()

    # For each item, verify label matches class_to_idx
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        img_name = dataset.filenames[idx]
        class_name = dataset.data[img_name]["name"]
        expected_label = dataset.class_to_idx[class_name]
        assert label == expected_label, f"Label mismatch for {img_name}: {label} != {expected_label}"