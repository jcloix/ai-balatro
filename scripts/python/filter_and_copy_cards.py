import os
import json
import shutil
import argparse


LABELS_PATH = "data/dataset_default/labels.json"  # your current JSON
OUTPUT_DIR = "data/dataset_base_only"  # your current JSON

def filter_and_copy(json_path, attribute, attribute_value, output_dir):
    # Load JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Prepare output dirs
    os.makedirs(output_dir, exist_ok=True)
    output_json_path = os.path.join(output_dir, "labels.json")

    # Filter cards matching attribute=value
    filtered = {}
    count = 0

    for filename, info in data.items():
        if info.get(attribute) == attribute_value:
            src = info.get("full_path")
            if not src or not os.path.isfile(src):
                print(f"⚠️ Skipping missing file: {src}")
                continue

            dst = os.path.join(output_dir, os.path.basename(src))
            shutil.copy2(src, dst)

            # Update path in copied JSON
            new_info = dict(info)
            new_info["full_path"] = dst
            filtered[filename] = new_info
            count += 1

    # Write new JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done. Copied {count} cards where {attribute}='{attribute_value}'.")
    print(f"📁 Output directory: {output_dir}")
    print(f"🗂  New JSON written to: {output_json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter cards by attribute=value and copy them with updated JSON."
    )
    parser.add_argument("--json", default=LABELS_PATH, help="Path to source JSON file.")
    parser.add_argument("--attribute", required=True, help="Attribute to filter by (e.g., modifier, rarity).")
    parser.add_argument("--attribute-value", required=True, help="Value of the attribute to match (e.g., Base).")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory for filtered dataset.")
    args = parser.parse_args()

    filter_and_copy(args.json, args.attribute, args.attribute_value, args.output_dir)


if __name__ == "__main__":
    main()
