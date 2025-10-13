# utils/analyze/dataset_stats.py
import json
import argparse
from collections import Counter

LABELS_JSON = "data/dataset_base_only/labels.json"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-json", type=str, default=LABELS_JSON)
    parser.add_argument("--top-n", type=int, default=20, help="Number of top classes to display")
    args = parser.parse_args()

    with open(args.labels_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    counts = Counter()
    for img_file, labels in data.items():
        counts[labels.get("name")] += 1

    sorted_counts = counts.most_common(args.top_n)
    print(f"Class distribution (top {args.top_n}):")
    for cls, c in sorted_counts:
        print(f"{cls:20} : {c}")
    print("\nTotal classes:", len(counts))

if __name__ == "__main__":
    main()
