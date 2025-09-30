import torch
import pytest
import numpy as np
from train_model import metrics

# -----------------------------
# Tests for Metrics class
# -----------------------------
def test_metrics_from_validation_and_epoch():
    val_metrics = metrics.Metrics.from_validation(val_loss=0.25, topk_acc=0.4, cm=np.array([[2,1],[0,3]]))
    assert val_metrics.val_loss == 0.25
    assert val_metrics.topk_acc == 0.4
    assert isinstance(val_metrics.cm, np.ndarray)

    epoch_metrics = metrics.Metrics.from_epoch(train_loss=0.1, val_metrics=val_metrics)
    assert epoch_metrics.train_loss == 0.1
    assert epoch_metrics.val_loss == val_metrics.val_loss
    assert epoch_metrics.topk_acc == val_metrics.topk_acc
    assert (epoch_metrics.cm == val_metrics.cm).all()

# -----------------------------
# Tests for top_k_accuracy
# -----------------------------
def test_top_k_accuracy():
    """
    Test the top_k_accuracy function with clear examples.
    """
    # 3 samples, 4 classes
    outputs = torch.tensor([
        [0.1, 0.8, 0.05, 0.05],  # top-1 = 1, top-2 = [1,0]
        [0.3, 0.4, 0.2, 0.1],    # top-1 = 1, top-2 = [1,0]
        [0.05, 0.2, 0.6, 0.15]   # top-1 = 2, top-2 = [2,1]
    ])
    labels = torch.tensor([1, 0, 2])  # true labels

    # Compute top-k accuracies
    acc1 = metrics.top_k_accuracy(outputs, labels, k=1)
    acc2 = metrics.top_k_accuracy(outputs, labels, k=2)
    acc3 = metrics.top_k_accuracy(outputs, labels, k=3)

    # Check expected values
    # sample-wise top-1 correct: [True, False, True] → 2/3 = 0.6667
    # top-2 correct: [True, True, True] → 3/3 = 1.0
    # top-3 correct: all 3 are within top-3 → 1.0
    assert pytest.approx(acc1, 0.01) == 2/3
    assert pytest.approx(acc2, 0.01) == 1.0
    assert pytest.approx(acc3, 0.01) == 1.0

# -----------------------------
# Tests for compute_confusion_matrix
# -----------------------------
class DummyModel(torch.nn.Module):
    def forward(self, x):
        # Return fixed logits
        return x

def test_compute_confusion_matrix(tmp_path):
    # Dummy dataset: 3 samples, 2 classes
    images = [torch.randn(1, 2) for _ in range(3)]
    labels = [torch.tensor([0]), torch.tensor([1]), torch.tensor([1])]
    loader = list(zip(images, labels))

    model = DummyModel()
    device = torch.device("cpu")
    num_classes = 2

    cm = metrics.compute_confusion_matrix(model, loader, device, num_classes)
    assert isinstance(cm, np.ndarray)
    assert cm.shape == (num_classes, num_classes)
    # Should sum to number of samples
    assert cm.sum() == 3
