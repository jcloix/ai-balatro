# test_dataset.py
import pytest
from PIL import Image
import torch
from torchvision import transforms
from train_model.dataset import CardDataset

# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def dummy_images(tmp_path):
    """Create 5 dummy PNG images."""
    paths = []
    for i in range(5):
        img_path = tmp_path / f"img{i+1}.png"
        img = Image.new("RGB", (32, 32), color=(i * 20, i * 20, i * 20))
        img.save(img_path)
        paths.append(str(img_path))
    return paths

@pytest.fixture
def labels_dict(dummy_images):
    """Return a labels dict using 'name' field."""
    return {path: {"name": f"class_{i}"} for i, path in enumerate(dummy_images)}

# -----------------------------
# Tests
# -----------------------------
def test_card_dataset_init(labels_dict):
    """Test __init__ and class/label mappings."""
    dataset = CardDataset(labels_dict)
    
    # Length
    assert len(dataset) == len(labels_dict)

    # class_names
    expected_classes = sorted({v["name"] for v in labels_dict.values()})
    assert dataset.class_names == expected_classes
    assert dataset.num_classes == len(expected_classes)
    
    # class_to_idx / idx_to_class
    for cls_name in expected_classes:
        idx = dataset.class_to_idx[cls_name]
        assert dataset.idx_to_class[idx] == cls_name

    # labels_dict contains integer labels
    for fname, label in dataset.labels_dict.items():
        class_name = labels_dict[fname]["name"]
        assert label == dataset.class_to_idx[class_name]

def test_card_dataset_getitem_and_len(labels_dict, dummy_images):
    """Test __getitem__ returns image and label."""
    dataset = CardDataset(labels_dict)
    
    # Access all items
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        assert isinstance(img, Image.Image)
        assert isinstance(label, int)
        assert 0 <= label < dataset.num_classes
    
    # Out-of-bounds raises IndexError
    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]

def test_card_dataset_transform_application(labels_dict):
    """Test transform is applied correctly."""
    dataset = CardDataset(labels_dict)
    
    # Apply ToTensor transform
    tensor_transform = transforms.ToTensor()
    dataset.transform = tensor_transform
    
    img, label = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape[0] == 3  # RGB channels
    assert isinstance(label, int)

def test_card_dataset_resume_classnames(labels_dict):
    """Test providing class_names manually (resume scenario)."""
    # Provide class_names in different order
    class_names = ["class_4", "class_0", "class_1", "class_3", "class_2"]
    dataset = CardDataset(labels_dict, class_names=class_names)
    
    # class_names and mapping should match provided list
    assert dataset.class_names == class_names
    assert dataset.num_classes == len(class_names)
    for i, cls in enumerate(class_names):
        assert dataset.class_to_idx[cls] == i
        assert dataset.idx_to_class[i] == cls

def test_card_dataset_with_full_path(tmp_path):
    """Test __getitem__ uses 'full_path' if provided."""
    # Create dummy images
    img_path = tmp_path / "img1.png"
    Image.new("RGB", (32, 32)).save(img_path)
    labels = {str(img_path): {"name": "A", "full_path": str(img_path)}}
    
    dataset = CardDataset(labels)
    img, label = dataset[0]
    assert isinstance(img, Image.Image)
    assert label == 0
