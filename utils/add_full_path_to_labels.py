import json
import os

# Paths
labels_path = "data/augmented.json"  # your current JSON
output_path = "data/augmented_with_path.json"
unlabeled_dir = "data/unlabeled"
augmented_dir = "data/augmented"

# Load existing labels
with open(labels_path, "r") as f:
    labels = json.load(f)

# Add full path to each label
updated_labels = {}
for fname, meta in labels.items():
    path_unlabeled = os.path.join(unlabeled_dir, fname).replace("\\", "/")
    path_augmented = os.path.join(augmented_dir, fname).replace("\\", "/")

    if os.path.exists(path_unlabeled):
        meta["full_path"] = path_unlabeled
    elif os.path.exists(path_augmented):
        meta["full_path"] = path_augmented
    else:
        print(f"Warning: {fname} not found in either folder, skipping")
        continue  # skip files not found

    updated_labels[fname] = meta

# Save updated JSON
with open(output_path, "w") as f:
    json.dump(updated_labels, f, indent=2)

print(f"Updated labels saved to {output_path}")
