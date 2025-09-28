import os
import tempfile
from PIL import Image
import matplotlib.pyplot as plt
from unittest import mock
import pytest

# This would be your visual check function refactored for testability
def display_augmented_images(original_dir, augmented_dir, max_augs=5, cards_per_row=3):
    original_images = [f for f in os.listdir(original_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    original_images.sort()
    rows = [original_images[i:i+cards_per_row] for i in range(0, len(original_images), cards_per_row)]

    for row_imgs in rows:
        fig, axes = plt.subplots(len(row_imgs), max_augs+1)
        if len(row_imgs) == 1:
            axes = [axes]
        for row_idx, orig_name in enumerate(row_imgs):
            name, ext = os.path.splitext(orig_name)
            orig_img = Image.open(os.path.join(original_dir, orig_name))
            # original image
            axes[row_idx][0].imshow(orig_img)
            # augmented images
            for i in range(max_augs):
                aug_file = f"{name}_aug{i}{ext}"
                aug_path = os.path.join(augmented_dir, aug_file)
                if os.path.exists(aug_path):
                    aug_img = Image.open(aug_path)
                    axes[row_idx][i+1].imshow(aug_img)
        plt.close(fig)

# Test
def test_visual_check_loads_images_correctly():
    with tempfile.TemporaryDirectory() as orig_dir, tempfile.TemporaryDirectory() as aug_dir:
        # create dummy original image
        orig_img_path = os.path.join(orig_dir, "card1.png")
        Image.new("RGBA", (32, 32), (255, 0, 0, 255)).save(orig_img_path)

        # create dummy augmented images
        for i in range(3):
            aug_img_path = os.path.join(aug_dir, f"card1_aug{i}.png")
            Image.new("RGBA", (32, 32), (0, 255, 0, 255)).save(aug_img_path)

        # Mock plt.show and plt.pause to avoid actual GUI popup
        with mock.patch("matplotlib.pyplot.show"), mock.patch("matplotlib.pyplot.pause"):
            display_augmented_images(orig_dir, aug_dir, max_augs=3, cards_per_row=1)
