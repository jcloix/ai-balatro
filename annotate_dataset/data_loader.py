# data_loader.py
import os
import json
from config.config import DATASET_DIR, LABELS_FILE, CARD_TYPES, CARDS_NAMES
from annotate_dataset.annotate_config import TRESHOLD_TOP2

import json
import os
from abc import ABC, abstractmethod

# --------------------------
# Base Class
# --------------------------
class BaseCardMaps(ABC):
    def __init__(self):
        self.prefill_map = {}   # Prefill values of a file
        self.group_map = {}     # Group very similar files together
        self.cluster_map = {}   # Group files that are a bit similar together
        self.file_map = {}      # Compute group and cluster id for a file.

    @abstractmethod
    def build(self, all_images):
        pass


# --------------------------
# Group/Cluster maps
# --------------------------
class GroupClusterMaps(BaseCardMaps):
    def __init__(self):
        super().__init__()

    def build(self, all_images):
        for f in all_images:
            cluster_id, group_id = parse_ids(f)
            self.group_map.setdefault(group_id, []).append(f)
            self.cluster_map.setdefault(cluster_id, []).append(f)
            self.file_map[f] = {
                "group_id": group_id,
                "cluster_id": cluster_id
            }
        return self


# --------------------------
# Inference maps
# --------------------------
class InferenceMaps(BaseCardMaps):
    def __init__(self, inference_file=f"{DATASET_DIR}/inference.json", reference_file=CARDS_NAMES):
        super().__init__()
        self.inference_file = inference_file
        self.reference_file = reference_file
        self.inference_data = {}
        self.reference_data = {}

        self._load_reference()
        self._load_inference()

    def _load_reference(self):
        if os.path.exists(self.reference_file):
            with open(self.reference_file, "r") as f:
                self.reference_data = json.load(f)
        else:
            raise ValueError(f"Reference file missing {self.reference_file}. Please execute ./script/compute_unique_cards.py")

    def _load_inference(self):
        if os.path.exists(self.inference_file):
            with open(self.inference_file, "r") as f:
                self.inference_data = json.load(f)
        else:
            raise ValueError(f"Inference file missing {self.inference_file}. Please execute python -m utils.inference_utils on the {DATASET_DIR}")

    def build(self, all_images):
        if not self.inference_data:
            return self  # no inference file, nothing filled

        for img in all_images:
            if img not in self.inference_data:
                raise ValueError(f"Inference missing for image: {img}")
            labels = self.inference_data[img]["labels"]["identification"]

            top1 = labels[0]["label"] if len(labels) > 0 else None
            top2 = labels[1]["label"] if len(labels) > 1 else None
            if not top1:
                raise ValueError(f"Inference top1 data missing for file: {img}")
            elif top1 not in self.reference_data:
                raise ValueError(f"Reference data missing for inferred card name: {top1}")
            
            # Group cards that should be the same (top1)
            self.group_map.setdefault(top1, []).append(img)
            if len(labels) > 1 and labels[1]["probability"] > TRESHOLD_TOP2:
                # Propose cards that might be the same (top2), in case the probability is good
                self.cluster_map.setdefault(top2, []).append(img)
            
            # Set file map
            self.file_map[img] = {
                "group_id": top1,
                "cluster_id":top2
            }

            card_info = self.reference_data[top1]
            # prefill map for current card
            self.prefill_map[img] = {
                "name": top1,
                "type": card_info.get("type"),
                "rarity": card_info.get("rarity"),
                "modifier": "Base"
            }

        return self

def build_maps(all_images, inference_file=f"{DATASET_DIR}/inference.json"):
    if os.path.exists(inference_file): # Use inference if it exist for semi-automatic annotate
        return InferenceMaps(inference_file=inference_file).build(all_images)
    return GroupClusterMaps().build(all_images)


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