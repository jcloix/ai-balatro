import json
import os

# Paths
labels_path = "data/dataset_default/labels.json"  # your current JSON
output_path = "data/dataset_default/labels_with_path.json"
unlabeled_dir = "data/dataset_default"

# Load existing labels
with open(labels_path, "r") as f:
    labels = json.load(f)

# Add full path to each label
updated_labels = {}
for fname, meta in labels.items():
    path_unlabeled = os.path.join(unlabeled_dir, fname).replace("\\", "/")

    if os.path.exists(path_unlabeled):
        meta["full_path"] = path_unlabeled
    else:
        print(f"Warning: {fname} not found in folder {path_unlabeled}, skipping")
        continue  # skip files not found

    updated_labels[fname] = meta

# Save updated JSON
with open(output_path, "w") as f:
    json.dump(updated_labels, f, indent=2)

print(f"Updated labels saved to {output_path}")
