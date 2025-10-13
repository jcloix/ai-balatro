import json
import argparse
from collections import defaultdict

LABELS_PATH = "data/dataset_default/labels.json"  # your current JSON

def find_missing_by_attribute(data, attribute):
    """Return per-name missing attribute values and all possible values."""
    name_to_attrs = defaultdict(set)
    all_values = set()

    for _, info in data.items():
        name = info.get("name")
        value = info.get(attribute)
        if name and value:
            name_to_attrs[name].add(value)
            all_values.add(value)

    missing = {}
    for name, values in name_to_attrs.items():
        missing_values = all_values - values
        if missing_values:
            missing[name] = sorted(missing_values)

    return missing, all_values


def find_missing_for_value(data, attribute, target_value):
    """Return all names that do NOT have the specified attribute value."""
    name_to_attrs = defaultdict(set)

    for _, info in data.items():
        name = info.get("name")
        value = info.get(attribute)
        if name and value:
            name_to_attrs[name].add(value)

    missing_names = [name for name, values in name_to_attrs.items() if target_value not in values]
    return sorted(missing_names)


def main():
    parser = argparse.ArgumentParser(description="Find missing attribute values for each unique name.")
    parser.add_argument("--json", default=LABELS_PATH, help="Path to JSON file.")
    parser.add_argument("--attribute", default="modifier", help="Attribute name to check (e.g., modifier, rarity).")
    parser.add_argument("--attribute-value", help="Optional: specific attribute value to check (e.g., Base).")
    args = parser.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.attribute_value:
        missing_names = find_missing_for_value(data, args.attribute, args.attribute_value)
        print(f"\n=== Names missing {args.attribute} = '{args.attribute_value}' ===")
        if not missing_names:
            print("✅ All names have this attribute value.")
        else:
            for name in missing_names:
                print(name)
    else:
        missing, all_values = find_missing_by_attribute(data, args.attribute)
        print(f"\n=== Unique {args.attribute} values found ===")
        print(sorted(all_values))

        print(f"\n=== Missing {args.attribute} values per name ===")
        if not missing:
            print("✅ No missing values — all names have full coverage!")
        else:
            for name, miss in missing.items():
                print(f"{name}: missing {miss}")


if __name__ == "__main__":
    main()
