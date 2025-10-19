# card_dataset/card.py
import os
import pyautogui
import imagehash
from build_dataset.build_config import OUTPUT_DIR, STRICT_DUP_THR, CANDIDATE_THR, CLUSTER_THR
from build_dataset.hash_storage import save_hashes, seen_hashes, cluster_counter, card_group_counter
from config.config import SCREEN_REGION, CARD_AREAS

def save_card(card_image):
    """Save card with hierarchical grouping."""
    global cluster_counter, card_group_counter

    hashes = {
        "ph": imagehash.phash(card_image),
        "ah": imagehash.average_hash(card_image),
        "dh": imagehash.dhash(card_image)
    }

    # find closest existing entry
    closest_idx = None
    min_distance = None
    for idx, h in enumerate(seen_hashes):
        ph_dist = hashes["ph"] - h["ph"]
        ah_dist = hashes["ah"] - h["ah"]
        dh_dist = hashes["dh"] - h["dh"]
        max_dist = max(ph_dist, ah_dist, dh_dist)
        if min_distance is None or max_dist < min_distance:
            min_distance = max_dist
            closest_idx = idx

    # decide thresholds
    if min_distance is not None:
        if min_distance <= STRICT_DUP_THR:
            matched_filename = seen_hashes[closest_idx].get("filename", f"card_{closest_idx+1}")
            return False, f"strict_duplicate_of:{matched_filename}"

        elif min_distance <= CANDIDATE_THR:
            cluster_id = seen_hashes[closest_idx]["cluster"]
            card_group_id = seen_hashes[closest_idx]["card_group"]
            label_cluster = cluster_id
            label_card_group = card_group_id

        elif min_distance <= CLUSTER_THR:
            cluster_id = seen_hashes[closest_idx]["cluster"]
            cluster_counter_val = cluster_id
            card_group_counter += 1
            card_group_id = card_group_counter
            label_cluster = cluster_id
            label_card_group = card_group_id

        else:
            cluster_counter += 1
            card_group_counter += 1
            label_cluster = cluster_counter
            label_card_group = card_group_counter
    else:
        cluster_counter += 1
        card_group_counter += 1
        label_cluster = cluster_counter
        label_card_group = card_group_counter

    # filename
    hash_prefix = str(hashes["ph"])[:8]
    filename = f"cluster{label_cluster}_card{label_card_group}_id{len(seen_hashes)+1}.png"
    card_image.save(os.path.join(OUTPUT_DIR, filename))

    seen_hashes.append({
        "ph": hashes["ph"],
        "ah": hashes["ah"],
        "dh": hashes["dh"],
        "cluster": label_cluster,
        "card_group": label_card_group,
        "filename": filename
    })
    save_hashes()
    return True, filename

def save_card_forced(card_index, card_areas, screen_region):
    screenshot = pyautogui.screenshot(region=screen_region)
    ax, ay, w, h = card_areas[card_index]
    card = screenshot.crop((ax, ay, ax + w, ay + h))
    hash_prefix = str(imagehash.phash(card))[:8]
    filename = f"forced_cluster0_card0_id{len(seen_hashes)+1}_{hash_prefix}.png"
    card.save(os.path.join(OUTPUT_DIR, filename))
    print(f"Force-saved card {card_index+1} as {filename}")
    return True, filename

# -----------------------------
# Capture wrapper
# -----------------------------
def capture_cards(show_only=False):
    screenshot = pyautogui.screenshot(region=SCREEN_REGION)
    if show_only:
        screenshot.show()
        return

    results = []
    for i, area in enumerate(CARD_AREAS):
        ax, ay, w, h = area
        card = screenshot.crop((ax, ay, ax + w, ay + h))
        unique, info = save_card(card)
        results.append((unique, info))
        if unique:
            print(f"Saved new card {i+1}: {info}")
        else:
            print(f"Card {i+1} duplicate of {info}")
    return results