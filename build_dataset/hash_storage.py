# card_dataset/hash_storage.py
import os
import json
from PIL import Image
import imagehash
from build_dataset.build_config import HASH_FILE

seen_hashes = []
cluster_counter = 0
card_group_counter = 0

def load_hashes():
    global seen_hashes, cluster_counter, card_group_counter
    if not os.path.exists(HASH_FILE):
        return

    with open(HASH_FILE, "r") as f:
        saved_data = json.load(f)
        for h in saved_data:
            card_group = h.get("card_group", h.get("cluster"))
            seen_hashes.append({
                "ph": imagehash.hex_to_hash(h["ph"]),
                "ah": imagehash.hex_to_hash(h["ah"]),
                "dh": imagehash.hex_to_hash(h["dh"]),
                "cluster": h["cluster"],
                "card_group": card_group,
                "filename": h.get("filename")
            })
        if seen_hashes:
            cluster_counter = max(h.get("cluster", 0) for h in seen_hashes)
            card_group_counter = max(h.get("card_group", 0) for h in seen_hashes)

def save_hashes():
    data = []
    for h in seen_hashes:
        data.append({
            "ph": str(h["ph"]),
            "ah": str(h["ah"]),
            "dh": str(h["dh"]),
            "cluster": h["cluster"],
            "card_group": h["card_group"],
            "filename": h.get("filename")
        })
    with open(HASH_FILE, "w") as f:
        json.dump(data, f, indent=2)
