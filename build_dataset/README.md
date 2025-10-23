# build_dataset Module

The `build_dataset` module is responsible for capturing and storing card images from the screen. It provides basic UI controls and utility functions to facilitate dataset creation, including pseudo-grouping and forced captures.

## Purpose

- Capture card images from a predefined screen region.
- Save captured cards in the `data/screenshots/` folder.
- Provide simple UI to visualize and verify capture regions.
- Support forced capture of individual cards with unique filenames.
- Provide **pseudo-grouping with hierarchical labels** to organize cards more accurately.
- Provide also perceptual hashes in "seen_hashes" variable to avoid too many duplicate

## Installation

Make sure your environment has the required packages:

```bash
pip install -r requirements.txt
```

## 🎯 Usage

### Launch the module

from build_dataset import build_dataset

```bash
python -m build_dataset.main  
```

The UI provides buttons and keyboard shortcuts:

| Action                     | Button / Key         |
|-----------------------------|-------------------|
| Show capture regions         | Show Regions (S)  |
| Hide capture regions         | Hide Regions (H)  |
| Capture screenshot           | Capture Screenshot (C) |
| Capture & save all cards     | Capture & Save (V) |
| Start automated capture      | Start Auto (A)    |
| Stop automated capture       | Stop Auto (X)     |
| Force save individual cards  | 1, 2, 3, 4        |

### Automatic capture

You can use the AHK script 'script/ahk/build_dataset.ahk' to automatise the capture of many screenshots in Balatro.
Hotkey: Ctrl + Shift + ;

## Hierarchical Pseudo-Grouping

Each captured card is analyzed using perceptual hashes (`phash`, `average_hash`, `dhash`).  
The module compares these hashes against previously saved cards to decide grouping:

- **Strict duplicate** → The card is considered a duplicate of an existing one and not saved again.
- **Candidate group** → The card belongs to the same **cluster** and **card group** as the closest match.
- **Cluster grouping** → The card shares the same cluster but starts a new card group.
- **New cluster** → The card is different enough to start a brand new cluster and card group.

### Labels in Filenames and JSON

Captured cards are saved with filenames reflecting their hierarchical grouping:

cluster{cluster_id}_card{card_group_id}_id{n}.png

Example:

cluster2_card5_id17.png

This means:
- Cluster ID: 2
-> If two cards share the same Cluster ID, it means the cards are similar. Allow to group similar cards together
and will help when labeling cards to check cards of same cluster.
- Card group ID: 5
-> If two cards share the same card group ID, it should in theory be the same card, but will still need manual check.
- Unique ID within the dataset: 17
-> Simply an ID to ensure uniqueness

The same hierarchical labels are also saved in the JSON metadata file alongside the hash values. Example entry:

{
  "filename": "cluster2_card5_id17.png",
  "cluster": 2,
  "card_group": 5,
  "ph": "7f2a3b4c5d6e7f80",
  "ah": "1234567890abcdef",
  "dh": "abcdef1234567890"
}

This JSON allows downstream modules like annotate_dataset and train_model to quickly access the grouping information and hashes for similarity checks.

## File Storage

All captured cards are stored in:

data/screenshots/

Filenames are automatically generated to avoid collisions, include a hash prefix, and reflect their **cluster and card group**.

## Notes

- The module does **not label cards** semantically; only pseudo-grouping is provided. Manual or downstream labeling should be done in the `label_dataset` module.
- Captures are limited to the region defined in `config.py`.
- Automated capture can be started and stopped using the UI or programmatically.

Since we have many **unique cards** (Jokers, Tarots, Planets, Spectrals), collecting
20–50 real images for each one would be too time-consuming. Don’t label all 175 unique cards at once:
- Start with a small subset (e.g. **10 Jokers + 5 Tarots**) and build the full pipeline.
- Once working, gradually add more cards.
- This avoids being blocked by the full dataset size.