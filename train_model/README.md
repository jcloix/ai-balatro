# train_model 🧠

The `train_model` module is part of the **Balatro** project.
Its purpose is to **train an AI model to recognize cards** based on the annotated dataset.
The trained model can later be used for real-time card identification with `identify_card` module.

---

## 🚀 Features

* Load annotated dataset created with `annotate_dataset`
* Train a model to recognize cards
* Support for detecting **identity cards** and **modifier variations**
* Save trained models for later inference
* Configurable training parameters (epochs, batch size, etc.)
* Configurable training strategy (Freeze, optimizer, scheduler, etc.)
* Scalable and flexible implementation of **task**
* Allow to train multi-head model or one head model
* Support checkpoints in case of failure or change of strategy for one model
* Add runtime augmentations to augment the dataset with noise
* Generic implementation of metrics to allow adding more in the future

---

## 🛠 Installation

Make sure you are inside the Balatro project folder and dependencies are installed:

```bash
pip install -r requirements.txt
```

---

## 🎯 Usage

Run the module directly:

```bash
python -m train_model.train
```

Or with arguments (See CLI section)

```bash
python -m train_model.train --tasks identification --epochs 7 --log-dir runs/identification --freeze-strategy all --optimizer simple --scheduler none --use-weighted-sampler
```

Monitoring can be don with Tensorboard

```bash
tensorboard --logdir runs/identification
```

### Example workflow

1. Prepare your dataset using:

   * `build_dataset` (capture images)
   * `annotate_dataset` (organize & label)
   * `augment_dataset` (Optionnally add offline augmentations)

2. Start training:

   ```bash
   python -m train_model --epochs 20 --batch-size 32
   ```

3. The trained model will be saved in the `data/models/` directory.

4. Use the trained model with `utils/inference_utils.py` for real-time recognition.

---

## Detailed features

## Balatro Train Module Features

The `train` module provides full functionality to train, validate, and monitor the card recognition model. Its main capabilities include:

### 1. Data Handling

* **Original/Augmented images:** Uses original and augmented labels for training.
* **Custom Dataset:** `CardDataset` loads images and labels from a dictionary or label files.
* **Train/Validation Split:** Split datasets into train/val sets with optional transforms.
* **Weighted Sampling:** Handle class imbalance with a `WeightedRandomSampler` if requested.
* **Dynamic Augmentation:** Apply train-time image augmentations (configurable via `Config.TRANSFORMS['train']`) to improve generalization.

### 2. Model Management

* **Pretrained Model:** Load a pretrained ResNet18 and replace the final layer to match the dataset’s number of classes.
* **Device Management:** Automatically uses GPU if available.
* **Freezing Backbone:** Optionally freeze feature extractor layers for fine-tuning only the classifier.

### 3. Training Pipeline

* **Forward & Backward Pass:** Handles normal and mixed-precision training automatically.
* **Validation:** Compute validation loss and optionally additional metrics during evaluation.

### 4. Metrics

* **Top-K Accuracy:** Compute Top-3 accuracy to measure how often the correct label is within the top-k predictions.
* **Confusion Matrix:** Compute class-by-class performance for detailed error analysis.
* **Confusion Summary:** Compute class-by-class performance summary for task with many classes.

### 5. Logging & Visualization

* **Console Logging:** Epoch statistics, including train/validation loss and optional metrics.
* **TensorBoard Logging:** Tracks loss, learning rate, and Top-K accuracy. Write to 'runs/' by default.

### 6. Checkpointing

* **Best Model Saving:** Automatically saves the model with the lowest validation loss.
* **Periodic Snapshots:** Saves checkpoints at regular intervals for safety.

### 7. Training Utilities

* **Early Stopping:** Monitors validation loss to stop training if performance stops improving.
* **Learning Rate Scheduling:** Supports StepLR or ReduceLROnPlateau depending on dataset size.
* **Resuming Training:** Load from a checkpoint to continue interrupted training.

### 8. Configurable Parameters

* **From `train_config.Config` :**

  * `EPOCHS` : Default number of training epochs
  * `BATCH_SIZE` : Default batch size
  * `LEARNING_RATE` : Initial learning rate
  * `VAL_SPLIT` : Default fraction of data for validation
  * `NUM_CLASSES` : Number of output classes
  * `TRANSFORMS` : Dictionary of image transformations (`train` / `test` / `none` / `heavy` / `light`) for dynamic augmentation

## ⚙️ Options

### General

* `--tasks` : Tasks to train (e.g. `identification`, `modifier`)
* `--epochs` : Number of training epochs (default from `Config.EPOCHS`)
* `--patience` : Early stopping patience
* `--log-dir` : Directory for TensorBoard logs (default `logs`)
* `--checkpoint-interval` : Frequency (in epochs) to save periodic snapshots
* `--resume` : Path to a checkpoint to resume training
* `--use-weighted-sampler` : Enable class-balanced sampling

### Strategy Selection

* `--freeze-strategy` : Freezing strategy (`none`, `high`, `mid`, `all`)
* `--optimizer` : Optimizer type (`simple`, `group`)
* `--scheduler` : Scheduler type (`cosine`, `plateau`, `step`, `none`)

### Optimizer Parameters

* `--optimizer-lr` : Learning rate for simple Adam
* `--optimizer-lr-backbone` : LR for backbone in GroupAdamW
* `--optimizer-lr-heads` : LR for heads in GroupAdamW
* `--optimizer-weight-decay` : Weight decay for GroupAdamW

### Scheduler Parameters

* `--scheduler-tmax` : T_max for CosineAnnealingLR
* `--scheduler-eta_min` : eta_min for CosineAnnealingLR
* `--scheduler-factor` : Factor for ReduceLROnPlateau
* `--scheduler-patience` : Patience for ReduceLROnPlateau
* `--scheduler-step-size` : Step size for StepLR
* `--scheduler-gamma` : Gamma for StepLR

---

## 🔄 Workflow Context

```text
build_dataset  --->  annotate_dataset  --->  train_model  --->  identify_card
```

---

# Data Collection & Training Strategy Decisions – Balatro

This document outlines the decisions and trade-offs for building the dataset of cards and training an AI model effectively. The goal is to balance realism, efficiency, and robustness.

---

## Training strategies

### 1. Framework Choice

**Option: PyTorch**

* **Advantages**:

  * Flexible, great for custom DataLoaders (fits Balatro’s dataset structure).
  * TorchVision has strong support for augmentations.
  * Widely used for research and experimentation.
* **Constraints**:

  * Slightly less “plug-and-play” than TensorFlow/Keras.

**Option: TensorFlow / Keras**

* **Advantages**:

  * High-level APIs (`model.fit`) make training easy.
  * Strong visualization/debugging tools (TensorBoard).
* **Constraints**:

  * Harder integration with custom dataset structures.
  * Less flexible for embedding/similarity-based experiments.

✅ **Decision**: Use **PyTorch** for modularity and flexibility.

---

### 2. Image Size & Preprocessing

**Option: Resize to fixed size (224×224)**

* **Advantages**: Matches pretrained model input; simple pipeline.
* **Constraints**: Minor distortion for non-square cards.

**Option: Keep native aspect ratio, pad to square**

* **Advantages**: Preserves exact proportions.
* **Constraints**: More complex preprocessing; still resized for pretrained models.

✅ **Decision**: Resize all images to **224×224**, normalized to **ImageNet mean/std**.

---

## Summary

* **Data**: Begin with collection screen captures, expand with gameplay samples.
* **Framework**: PyTorch.
* **Preprocessing**: 224×224, ImageNet normalization.
* **Augmentation**: Hybrid (offline + runtime, no flips).
* **Training defaults**: Batch 32, LR 1e-4, Adam, ~20 epochs.
* **Regularization**: Early stopping + checkpointing.

---

## 📌 Notes

* The `train_model` module centralizes the full Balatro training workflow.
* All CLI arguments are parsed and documented in `train.py`.
* Compatible with **single-head** and **multi-head** models (e.g., identification, modifier).
* Automatically handles checkpoints, scheduler steps, and early stopping.
* Designed for flexible experimentation with consistent and reproducible setups.
* Monitored through TensorBoard for live visualization of metrics and loss curves.
