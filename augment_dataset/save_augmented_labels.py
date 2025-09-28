import os
import json
import re
from config.config import DATASET_AUGMENTED_DIR, AUGMENTED_LABELS_FILE, LABELS_FILE


def load_original_labels():
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def find_parent(filename):
    # Matches pattern like: cluster5_card17_id302_aug3.png -> cluster5_card17_id302.png
    match = re.match(r"(.*)(_aug\d+)(\.\w+)$", filename)
    if match:
        parent_name = match.group(1) + match.group(3)
        augmentation_type = match.group(2).lstrip("_")
        return parent_name, augmentation_type
    return None, None

def build_augmented_labels(original_labels):
    augmented_labels = {}
    for fname in os.listdir(DATASET_AUGMENTED_DIR):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        parent, aug_type = find_parent(fname)
        if parent is None or parent not in original_labels:
            print(f"Warning: could not find parent for {fname}")
            continue
        # Copy parent's metadata
        meta = original_labels[parent].copy()
        meta["parent"] = parent
        meta["augmentation"] = aug_type
        augmented_labels[fname] = meta
    return augmented_labels

def main():
    original_labels = load_original_labels()
    augmented_labels = build_augmented_labels(original_labels)
    os.makedirs(os.path.dirname(AUGMENTED_LABELS_FILE), exist_ok=True)
    with open(AUGMENTED_LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(augmented_labels, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(augmented_labels)} augmented labels to {AUGMENTED_LABELS_FILE}")

if __name__ == "__main__":
    main()
