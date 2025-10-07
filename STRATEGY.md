# 🃏 Balatro — Augmentation & Training Strategy

This README describes a **complete strategy** for training your shared multi-head Balatro model (card name + modifier). It explains how to handle **data imbalance**, **offline augmentations**, and **training best practices** for reliable convergence.

---

## 📦 Dataset Overview

* Each card has one **main image** and possibly multiple duplicates.
* Some cards appear **up to 15×**, others only **once**.
* **Modifiers**:

  * `Base` → default, no visual effect (most common)
  * `Foil`, `Holographic`, `Polychrome` → mild filter effects
  * `Negative` → strong color inversion, very rare (~50 total)

**Goal:** Train a single model with two heads:

* **Head A:** predicts the **card name** (identity)
* **Head B:** predicts the **modifier**

---

## 🧩 1. Offline Augmentation (recommended)

Run a preprocessing script to create augmented variants of your dataset. This reduces imbalance and increases robustness.

### 🪄 Strategy

* Cards with **few samples (< 3)** → generate **10 augmentations** each.
* Cards with **more samples** → generate **3 augmentations**.
* For modifiers:

  * `Base` → full augmentation pipeline.
  * `Foil/Holographic/Polychrome` → light geometric transforms only.
  * `Negative` → usually skip or apply minimal jitter (don’t distort).
* Optionally synthesize new **Negative** samples by inverting `Base` cards.

### Example Config

```python
CFG = {
    "DATA_DIR": "data/cards",
    "AUG_DIR": "data/augmented",
    "LABELS_FILE": "data/cards/labels.json",
    "AUG_JSON": "data/augmented/augmented.json",
    "IMG_SIZE": 224,
    "STANDARD_AUG_PER_SAMPLE": 3,
    "RARE_AUG_PER_SAMPLE": 10,
    "RARE_THRESHOLD": 3,
    "NEGATIVE_SYNTHETIC": True,
}
```

### Example Transform Ideas

* **Identity-safe:** rotation (±10°), small crops, flips.
* **Color-safe:** low brightness/contrast jitter.
* **Negative synthetic:** invert RGB channels of `Base` cards.

Run the generator to produce offline augmentations and metadata (JSON with `synthetic` flag).

---

## 🔁 2. Online (Runtime) Transforms

Use lightweight runtime transforms during training:

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(8),
    transforms.ColorJitter(0.08, 0.08, 0.04),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
```

---

## ⚖️ 3. Sampling and Balancing

Since you have strong imbalance (e.g., many `Base`, few `Negative`), use **WeightedRandomSampler** or **per-head balancing**.

### Recommended:

* Compute weights inversely proportional to class frequency.
* Use the same sampler for both heads to keep labels aligned.

Example:

```python
sampler = WeightedRandomSampler(weights, num_samples=len(weights))
train_loader = DataLoader(dataset, sampler=sampler, batch_size=32)
```

---

## 🧠 4. Model Training Setup

### Model Architecture

* **Shared Encoder:** CNN backbone (e.g., ResNet50, EfficientNet)
* **Two Heads:**

  * Head 1 → `card_name` classifier (e.g., 200–300 classes)
  * Head 2 → `modifier` classifier (5 classes: Base, Foil, Holo, Poly, Negative)

### Loss Function

Use weighted cross-entropy for both heads:

```python
loss_name = F.cross_entropy(name_logits, name_labels, weight=name_class_weights)
loss_mod  = F.cross_entropy(mod_logits, mod_labels, weight=modifier_class_weights)
loss = loss_name + 0.5 * loss_mod   # optionally weight less frequent head
```

### Optimizer & Scheduler

* Optimizer: `AdamW(lr=1e-4, weight_decay=1e-4)`
* Scheduler: `CosineAnnealingLR` or `OneCycleLR`
* Early stopping on validation loss or accuracy

### Training Loop Summary

1. Merge original and augmented JSONs.
2. Build train/val split by **card name**, ensuring same card doesn’t appear in both.
3. Apply runtime transforms.
4. Train 30–50 epochs with mixed precision (AMP).
5. Log per-head accuracies.

---

## 📊 5. Evaluation

Metrics per head:

* **Card name:** top-1 and top-3 accuracy
* **Modifier:** accuracy and confusion matrix
* Track performance on rare modifiers (`Negative`, `Foil`, etc.) separately.

Consider saving embeddings from the shared encoder for clustering analysis (helps visualize if modifiers are separable in feature space).

---

## 🧩 6. Validation Best Practices

To avoid data leakage:

* Split **by card identity**, not by image file.
* Ensure augmented variants of a card only exist in **one split**.

---

## 🚀 7. Optional Extensions

* Use **MixUp/CutMix** for further regularization (helpful for overrepresented classes).
* Implement **curriculum learning** (start with Base-only training, then fine-tune on modifiers).
* Save feature maps for interpretability (Grad-CAM on modifiers).

---

## ✅ Summary

| Step | Component            | Purpose                       |
| ---- | -------------------- | ----------------------------- |
| 1    | Offline Augmentation | Balance dataset & add variety |
| 2    | Runtime Transforms   | Improve generalization        |
| 3    | Weighted Sampling    | Handle imbalance              |
| 4    | Multi-head Loss      | Train card + modifier jointly |
| 5    | Validation by Card   | Prevent data leakage          |

---

## 📁 Suggested Folder Structure

```
project/
├─ data/
│  ├─ cards/
│  │   ├─ labels.json
│  ├─ augmented/
│  │   ├─ augmented.json
│  │   ├─ *.png
├─ config/
│  └─ augmentation_config.py
├─ scripts/
│  └─ generate_offline_augmentations.py
├─ train/
│  ├─ transforms.py
│  ├─ train_model.py
│  └─ dataset.py
└─ models/
   └─ multihead_model.py
```
