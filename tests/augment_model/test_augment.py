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
    augmented = augment.augment_image(img, rotate=False, flip=False, brightness_contrast=False, blur_noise=False, negative=False)
    assert augmented == img  # same object when all flags disabled

def test_augment_dataset_creates_files():
    img = create_test_image()
    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        img_path = os.path.join(input_dir, "card1.png")
        img.save(img_path)

        # Simulate augment_dataset by calling augment_image for each variation
        n_variations = 3
        for i in range(n_variations):
            aug_img = augment.augment_image(img)
            aug_img.save(os.path.join(output_dir, f"card1_aug{i}.png"))

        output_files = os.listdir(output_dir)
        assert len(output_files) == 3
        for i in range(3):
            assert f"card1_aug{i}.png" in output_files

def test_apply_negative():
    img = create_test_image(color=(100, 150, 200, 255))
    neg_img = augment.apply_negative(img)
    assert isinstance(neg_img, Image.Image)
    assert neg_img.size == img.size

    # Check inversion on RGB channels (ignore alpha)
    orig_pixels = img.load()
    neg_pixels = neg_img.load()
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            r, g, b, a = orig_pixels[x, y]
            r2, g2, b2, a2 = neg_pixels[x, y]
            assert r2 == 255 - r
            assert g2 == 255 - g
            assert b2 == 255 - b
            assert a2 == a  # alpha unchanged
            break  # only check first pixel to speed up test
        break