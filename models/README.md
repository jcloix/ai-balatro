# models

This module defines and exposes model architectures for import by other modules.

It includes:

* **MultiHeadModel:** A ResNet18 backbone with multiple output heads for multi-task classification.
* **build_model:** Builds a single-head ResNet18 classifier.
* **build_multi_model:** Builds a multi-head version for multiple tasks.
* **load_checkpoint:** Loads model weights from a checkpoint.
* **unwrap_model:** Helper for running inference and extracting outputs.

Purpose: Centralize model definitions so that training, evaluation, and inference modules can easily import and reuse them.
