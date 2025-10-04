import json
import os

# Paths
labels_path = "data/dataset_default/labels.json"  # your current JSON
output_path = "data/reference_data/cards_names.json"

# Load existing labels
with open(labels_path, "r") as f:
    labels = json.load(f)

# Collect unique names
unique_cards = {}

for _, card in labels.items():
    name = card["name"]
    # Only keep one type/rarity per name
    if name not in unique_cards:
        unique_cards[name] = {
            "type": card["type"],
            "rarity": card["rarity"]
        }

# Save updated JSON
with open(output_path, "w") as f:
    json.dump(unique_cards, f, indent=2)

print(f"Saved unique cards to {output_path}")
