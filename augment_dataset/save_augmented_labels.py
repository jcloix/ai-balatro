import os
import json
import re
import argparse


def load_original_labels(labels_file):
    with open(labels_file, "r", encoding="utf-8") as f:
        return json.load(f)


def find_parent(filename):
    """
    Matches pattern like:
    cluster5_card17_id302_aug3.png -> cluster5_card17_id302.png
    Returns (parent_name, augmentation_type)
    """
    match = re.match(r"(.*)(_aug\d+)(\.\w+)$", filename)
    if match:
        parent_name = match.group(1) + match.group(3)
        augmentation_type = match.group(2).lstrip("_")
        return parent_name, augmentation_type
    return None, None


def build_augmented_labels(augmented_dir, original_labels):
    augmented_labels = {}
    for fname in os.listdir(augmented_dir):
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
        # Update full_path to the current augmented file
        meta["full_path"] = os.path.join(augmented_dir, fname).replace("\\", "/")
        augmented_labels[fname] = meta
    return augmented_labels


def main():
    parser = argparse.ArgumentParser(description="Generate augmented labels JSON")
    parser.add_argument(
        "--aug-dir", type=str, required=True,
        help="Directory containing augmented images"
    )
    parser.add_argument(
        "--labels", type=str, required=True,
        help="Original labels.json file"
    )
    args = parser.parse_args()

    augmented_dir = args.aug_dir
    labels_file = args.labels

    original_labels = load_original_labels(labels_file)
    augmented_labels = build_augmented_labels(augmented_dir, original_labels)

    # Save augmented.json inside augmented_dir
    output_file = os.path.join(augmented_dir, "augmented.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(augmented_labels, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(augmented_labels)} augmented labels to {output_file}")


if __name__ == "__main__":
    main()
