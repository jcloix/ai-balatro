import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import pytest
from unittest.mock import MagicMock, patch

from train_model import train_loops

# --------------------------
# Fixtures
# --------------------------
@pytest.fixture
def dummy_model():
    return nn.Linear(10, 3)

@pytest.fixture
def small_dataset():
    X = torch.randn(6, 10)
    y = torch.randint(0, 3, (6,))
    return TensorDataset(X, y)

@pytest.fixture
def small_loader(small_dataset):
    return DataLoader(small_dataset, batch_size=2)

# --------------------------
# Forward / Backward Tests
# --------------------------
def test_forward_backward_no_scaler(dummy_model):
    x = torch.randn(2, 10)
    y = torch.randint(0, 3, (2,))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(dummy_model.parameters())

    outputs, loss = train_loops.forward_pass(dummy_model, x, y, criterion)
    train_loops.backward_step(loss, optimizer)

    assert isinstance(outputs, torch.Tensor)
    assert loss.item() >= 0
    for p in dummy_model.parameters():
        assert p.grad is not None

def test_forward_backward_with_dummy_scaler(dummy_model):
    x = torch.randn(2, 10)
    y = torch.randint(0, 3, (2,))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(dummy_model.parameters())
    scaler = MagicMock()

    outputs, loss = train_loops.forward_pass(dummy_model, x, y, criterion, scaler=scaler)
    train_loops.backward_step(loss, optimizer, scaler=scaler)

    # Ensure scaler methods are called
    scaler.scale.assert_called_with(loss)
    scaler.step.assert_called_with(optimizer)
    scaler.update.assert_called()

# --------------------------
# Training / Validation Tests
# --------------------------
def test_train_one_epoch_returns_loss(dummy_model, small_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(dummy_model.parameters())
    device = torch.device("cpu")

    loss = train_loops.train_one_epoch(dummy_model, small_loader, criterion, optimizer, device)
    assert loss >= 0

def test_validate_returns_metrics(dummy_model, small_loader):
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    # Compute metrics
    metrics = train_loops.validate(dummy_model, small_loader, criterion, device, compute_metrics=True, num_classes=3)

    # Check metrics attributes
    assert metrics.val_loss >= 0
    assert metrics.topk_acc is not None
    assert metrics.cm.shape == (3, 3)

def test_validate_no_metrics(dummy_model, small_loader):
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    metrics = train_loops.validate(dummy_model, small_loader, criterion, device, compute_metrics=False)
    assert metrics.val_loss >= 0
    assert metrics.topk_acc is None
    assert metrics.cm is None

# --------------------------
# Edge Case Tests
# --------------------------
def test_train_one_epoch_empty_loader(dummy_model):
    empty_loader = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(dummy_model.parameters())
    device = torch.device("cpu")

    # Should handle gracefully without error
    loss = train_loops.train_one_epoch(dummy_model, empty_loader, criterion, optimizer, device)
    assert loss == 0 or loss is None  # depending on implementation

def test_validate_empty_loader(dummy_model):
    empty_loader = []
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    metrics = train_loops.validate(dummy_model, empty_loader, criterion, device, compute_metrics=True, num_classes=3)
    assert metrics.val_loss == 0 or metrics.val_loss is None
    assert metrics.topk_acc is None or metrics.topk_acc == 0
    assert metrics.cm is None
