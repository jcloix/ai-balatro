import pytest
import os
import tempfile
from PIL import Image
import augment_dataset.augment as augment

def create_test_image(size=(64, 64), color=(255, 0, 0, 255)):
    """Create a simple RGBA image for testing."""
    img = Image.new("RGBA", size, color)
    return img

def test_rotate_image():
    img = create_test_image()
    rotated = augment.rotate_image(img, max_angle=10)
    assert isinstance(rotated, Image.Image)
    assert rotated.size[0] >= img.size[0]  # expanded due to rotation
    assert rotated.size[1] >= img.size[1]

def test_flip_image():
    img = create_test_image()
    flipped = augment.flip_image(img)
    assert isinstance(flipped, Image.Image)
    assert flipped.size == img.size

def test_adjust_brightness_contrast():
    img = create_test_image()
    adjusted = augment.adjust_brightness_contrast(img, brightness_range=(0.9, 1.1), contrast_range=(0.9, 1.1))
    assert isinstance(adjusted, Image.Image)
    assert adjusted.size == img.size

def test_add_blur_noise():
    img = create_test_image()
    result = augment.add_blur_noise(img, blur_prob=1.0, noise_prob=1.0)  # force both
    assert isinstance(result, Image.Image)
    assert result.size == img.size

def test_augment_image_flags():
    img = create_test_image()
    # disable all to check image remains the same
    augmented = augment.augment_image(img, rotate=False, flip=False, brightness_contrast=False, blur_noise=False)
    assert augmented == img  # same object when all flags disabled

def test_augment_dataset_creates_files():
    img = create_test_image()
    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        img_path = os.path.join(input_dir, "card1.png")
        img.save(img_path)
        augment.augment_dataset(input_dir, output_dir, n_variations=3, seed=42)
        output_files = os.listdir(output_dir)
        assert len(output_files) == 3
        for i in range(3):
            assert f"card1_aug{i}.png" in output_files
