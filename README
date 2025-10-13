# Balatro 🃏

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![License](https://img.shields.io/badge/License-MIT-green)

Balatro is a project designed to learn and experiment with AI in a real scenario: identifying cards. From dataset creation to real-time card recognition, Balatro provides a full workflow with modular design.
Balatro performs instance-level card identification: the AI learns to recognize each unique card individually, rather than just its type. Each card is labeled by its unique ID, and augmented variations maintain this ID for training.

---

## 🚀 Features

- Capture card screenshots and organize a dataset
- Annotate cards with proper labels
- Train an AI model to identify cards and their modifiers
- Identify cards in real-time scenarios
- Useful utility functions for automated card detection

---

## 🗂 Project Modules

- **build_dataset**  
  Capture screenshots of cards with a minimal UI and pseudo-grouping functions.

- **annotate_dataset**  
  Organize and label the dataset properly. A more advanced UI may be implemented.

- **train_model**  
  Train AI models to identify cards and recognize polychrome.

- **models**  
  Models of the project to share it between the modules. Used to the MultiHeadModel

- **utils**  
  Helper functions, e.g., check if a specific `<card_id>` is displayed on the screen.

- **config**  
  Configuration settings and constants used throughout the project.

---

## 🛠 Installation

Make sure Python 3.10+ is installed. Then:

```bash
git clone https://github.com/yourusername/balatro.git
cd balatro
pip install -r requirements.txt
```

---

## 🎯 Usage

### 1️⃣ Build the dataset
```bash
python -m build_dataset
```

### 2️⃣ Annotate the dataset
```bash
python -m annotate_dataset
```

### 3️⃣ Train the AI
```bash
python -m train_model
```


### 5️⃣ Use helper functions
```python
from utils import is_card_displayed

if is_card_displayed("card_123"):
    print("Card 123 is currently on screen!")
```

---

## 🔄 Workflow Diagram


                        ┌────────────────────┐
                        │   build_dataset    │
                        │  (data capture)    │
                        │  [optional UI:     │
                        │   screenshot,      │
                        │   temp grouping]   │
                        └────────┬──────────┘
                                 │
                      Screenshots / Temporary Groups
                                 │
                        ┌────────▼──────────┐
                        │ annotate_dataset  │
                        │ (organize & label │
                        │  dataset)         │
                        │  [UI: drag & drop,│
                        │   assign labels,  │
                        │   verify]         │
                        └────────┬──────────┘
                                 │
                    Labeled & Annotated Dataset (CSV, JSON, etc.)
                                 │
                        ┌────────▼──────────┐
                        │ augment_dataset   │
                        │ (data augmentation│
                        │  & synthetic      │
                        │  variations)      │
                        │  [rotate, crop,   │
                        │   brightness, etc.]│
                        └────────┬──────────┘
                                 │
                    Augmented Dataset (ready for training)
                                 │
                        ┌────────▼──────────┐
                        │   train_model     │
                        │ (train AI on      │
                        │ augmented &       │
                        │ annotated data)   │
                        └────────┬──────────┘
                                 │
                        Trained Model Files
                                 │
                        ┌────────▼──────────┐
                        │  identify_card    │
                        │ (inference /      │
                        │ real-time use)    │
                        │  [optional UI:    │
                        │   highlight cards]│
                        └────────┬──────────┘
                                 │
                 Detected Cards / Modifiers Info
                                 │
                        ┌────────▼──────────┐
                        │      utils        │
                        │ (shared helpers)  │
                        │image preprocessing│
                        │,logging, file I/O │
                        └───────────────────┘


## Detailed Step-by-Step Workflow for training

### 1️⃣ Data Preparation 

1. **Capture raw screenshots**

   * Screenshots taken from real gameplay.
   * Stored in `data/unlabeled/`.

2. **Annotate dataset**

   * Use `annotate_dataset` to label each card.
   * Labels stored in `data/dataset_default/labels.json`.
   * (Optional) the inference can help pre-fill the fields

3. **Optional offline data augmentation**

   * Apply transformations (rotation, brightness, cropping, etc.).
   * Augmented images stored in `data/augmented/`.
   * Metadata stored in `data/augmented.json`.


---

### 2️⃣ Load existing checkpoint if requested 

1. Load existing checkpoint if any 

   * Loads checkpoint if provided to continue the training of an existing model
   * The checkpoints contain the following
   ```python
   checkpoint = {
        "epoch": epoch,
        "model": training_state.model.state_dict(),
        "optimizer": training_state.optimizer.state_dict(),
        "scheduler": training_state.scheduler.state_dict() if training_state.scheduler else None,
        "scaler": training_state.scaler.state_dict() if training_state.scaler else None,
        "head_states": {}
        "head_states":{
            "identification":{
               "class_names":170
            },
            "modifier":{
               "class_names":5
            }
        }
    }
   ```


### 3️⃣ Dataset Loading

1. Prepare the correct task(s) to execute

   * Each task represent a goal for the model to learn, by default it is "identification" and "modifier"
   * When multi tasks are provided, the model will have multiple heads with each task

2. Initialize the strategy to use

   * Prepare the strategy (frozen, ptimizer and scheduler) to be use for the training
   * Strategy can be easily configurable via run arguments

3. Initialize `CardDataset`

   * Loads image paths + labels from the merged JSON mapping in memory.
   * Provide the image paths that will be loaded and transformed on-the-fly when accessed by the DataLoader during training or validation.
   * Define the image transforms (augmentation, normalization, resizing) that will be applied later for training and validation.

4. Split into training and validation sets

   * Split the dataset into training and validation subsets using `Config.VAL_SPLIT`.
   * Create DataLoaders for each subset:

     * Train loader → shuffled, batched.
     * Validation loader → not shuffled, batched.

5. **Optional Weighted Sampler**

   * If `use_weighted_sampler` is enabled, balance class distribution in the training loader.

---

### 4️⃣ Initialize Model and Training Parameters

1. **Load pretrained ResNet18 (convolutional neural network)**

   * `resnet18(pretrained=True)`

2. **Replace final layer**

   * Replace ResNet18’s original final layer (1000 outputs) with a new linear layer:

```python
model.fc = nn.Linear(num_features, NUM_CLASSES)
```

* `num_features` → input features to the layer (512 for ResNet18)
* `NUM_CLASSES` → number of unique cards in our dataset
* This allows the model to output predictions for our specific card classes instead of ImageNet classes.

3. **Send model to device**

   * GPU if available (`cuda`), else CPU

4. **Freeze Backbone (optional)**

   * If `freeze_backbone` is enabled, freeze all layers except the final fully connected layer.

5. **Setup training parameters**

   * **Loss function:** `CrossEntropyLoss`
   * **Optimizer:** `Adam` with learning rate from config/arguments
   * **Scheduler:** StepLR or ReduceLROnPlateau depending on dataset size
   * **Mixed precision scaler:** `GradScaler` for faster GPU training (optional)
   * **Early stopping:** Monitors validation loss to stop training if no improvement
   * **TensorBoard writer:** Logs metrics for visualization

---

### 5️⃣ Training Loop

For each epoch:

1. **Train one epoch per head**

   * Iterate through training loader batches.
   * Move images + labels to device (GPU or CPU).
   * Forward pass → compute predictions.
   * Compute loss (`CrossEntropyLoss`).
   * Backward pass → compute gradients.
   * Optimizer step → update weights.
   * Compute average training loss.

2. **Validate per head**

   * Iterate through validation loader batches.
   * Forward pass only (no gradients).
   * Compute validation loss and metrics (Top-K accuracy, confusion matrix).
   * Average validation loss.

3. **Logging per head**

   * Print epoch stats (train_loss, val_loss, metrics).
   * Optional: log to TensorBoard.

4. **Learning rate scheduling**

   * Adjust the learning rate during training (StepLR or ReduceLROnPlateau).

5. **Checkpointing**

   * Save best model if val loss improves (`best_model.pth`).
   * Optional: periodic checkpoints.

6. **Early stopping**

   * Track consecutive epochs without improvement.
   * Stop if validation loss doesn’t improve for `patience` epochs.

---

### 6️⃣ End of Training

1. Save final model (optional).
2. Model ready for inference, embedding extraction, or further fine-tuning.

---

### Key Notes

* **Transforms**: offline + on-the-fly augmentations.
* **Early stopping + checkpointing** prevents overfitting.
* **Validation set** ensures generalization.
* **Modular design** separates dataset, model, and training steps for clarity and extendability.


## 📦 Handling of Data & Metadata

Balatro keeps a strict separation between **raw data**, **annotations**, **augmentations**, and **training sets**. This ensures clarity and reproducibility.

### 1. **Raw Data**

```
data/screenshots/*.png
data/screenshots/seen_hashes.json
```

* All collected screenshots from gameplay.
* `seen_hashes.json` stores perceptual hashes and cluster assignments to avoid duplicates.
* This folder is never modified after creation.

---

### 2. **Annotations**

```
data/dataset_default/labels.json
```
* Data are moved into data/cards
* Human-provided labels for each raw image.
* Maps original filenames → card identity (`name`, `type`, `rarity`, `modifier`, …).
* Example:

  ```json
  "cluster5_card17_id302.png": {
    "name": "Dna",
    "type": "Joker",
    "rarity": "Rare",
    "modifier": "Base",
    "cluster": 5,
    "card_group": 17
  }
  ```
* This file only covers **original images**, not augmented ones.
* Serves as the single source of truth for manual annotations.

---

### 3. **Augmented Images & Metadata**

```
data/augmented/*.png
data/augmented/augmented.json
```

* `data/augmented/` contains only augmented images.
* `data/augmented.json` contains metadata for the augmented images, referencing their original parent.
* Example:

  ```json
  "cluster5_card17_id302_aug3.png": {
    "name": "Dna",
    "type": "Joker",
    "rarity": "Rare",
    "modifier": "Base",
    "cluster": 5,
    "card_group": 17,
    "parent": "cluster5_card17_id302.png",
    "augmentation": "rotate+15"
  }
  ```
* Automatically generated by `augment_dataset`.
* If you use a small subset for testing (e.g., `data/test_subset/`), `data/augmented/` will contain only augmented images from that subset.
* No separate `data/test_augmented/` folder is required.

---

---

### 🔑 Key Principles

* **Raw stays raw** → `data/screenshots` contains unlabeled images only.
* **Metadata centralized** → Move originals cards into `data/dataset_default`. JSON original labels: `data/dataset_default/labels.json`, augmentations: `data/augmented/augmented.json`.

---



## 🤝 Contributing

Open issues, submit pull requests, suggest improvements for UI and AI accuracy.

---

## 📄 License

MIT License

Copyright (c) 2025 Julien Cloix

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


