import os
from PIL import Image
import matplotlib.pyplot as plt
from config.config import DATASET_DIR, DATASET_AUGMENTED_DIR

# Folders

# Max augmentations to display per original
max_augs = 5

# How many cards per figure (row-wise)
cards_per_row = 3

# Time to display each figure (seconds)
display_time = 3

# List original images
original_images = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
original_images.sort()

# Prepare rows
rows = []
for i in range(0, len(original_images), cards_per_row):
    rows.append(original_images[i:i+cards_per_row])

for row_imgs in rows:
    fig, axes = plt.subplots(len(row_imgs), max_augs+1, figsize=(3*(max_augs+1), 4*len(row_imgs)))
    
    # Ensure axes is 2D list
    if len(row_imgs) == 1:
        axes = [axes]
    
    for row_idx, orig_name in enumerate(row_imgs):
        name, ext = os.path.splitext(orig_name)
        # Original image
        orig_img = Image.open(os.path.join(DATASET_DIR, orig_name))
        axes[row_idx][0].imshow(orig_img)
        axes[row_idx][0].axis("off")
        axes[row_idx][0].set_title("Original")
        
        # Augmented images
        for i in range(max_augs):
            aug_file = f"{name}_aug{i}{ext}"
            aug_path = os.path.join(DATASET_AUGMENTED_DIR, aug_file)
            if os.path.exists(aug_path):
                aug_img = Image.open(aug_path)
                axes[row_idx][i+1].imshow(aug_img)
                axes[row_idx][i+1].axis("off")
                axes[row_idx][i+1].set_title(f"Aug {i}")
            else:
                axes[row_idx][i+1].axis("off")
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(display_time)  # show for display_time seconds
    plt.close(fig)
