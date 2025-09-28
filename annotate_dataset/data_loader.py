# data_loader.py
import os
import json
from config.config import DATASET_DIR, LABELS_FILE, CARD_TYPES

def load_labels():
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            return json.load(f)
    return {}

def list_images():
    return [f for f in os.listdir(DATASET_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

def parse_ids(filename):
    parts = filename.split("_")
    cluster_id = int(parts[0].replace("cluster", ""))
    card_group_id = int(parts[1].replace("card", ""))
    return cluster_id, card_group_id

def build_maps(all_images):
    card_group_map = {}
    cluster_map = {}
    for f in all_images:
        cluster_id, card_group_id = parse_ids(f)
        card_group_map.setdefault(card_group_id, []).append(f)
        cluster_map.setdefault(cluster_id, []).append(f)
    return card_group_map, cluster_map

def get_unlabeled_groups(card_group_map, labels):
    unlabeled_card_groups = [
        cg for cg, files in card_group_map.items()
        if any(f not in labels for f in files)
    ]
    unlabeled_card_groups.sort()
    return unlabeled_card_groups

def compute_unique_by_type(labels):
    unique_by_type = {t: set() for t in CARD_TYPES}
    for v in labels.values():
        if "name" in v and "type" in v:
            unique_by_type[v["type"]].add(v["name"])
    return unique_by_type