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
# Dummy wrappers
# --------------------------
class DummyHead:
    def __init__(self, loader):
        self.train_loader = loader
        self.val_loader = loader
        self.criterion = nn.CrossEntropyLoss()
        self.name = "dummy"
        self.metrics = None  # or mock if needed

class DummyState:
    def __init__(self, model, optimizer, device="cpu", scaler=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler

class DummyEpochResult:
    def set_train(self, train_loss):
        self.train_loss = train_loss
    def set_val(self, val_loss, val_outputs=None, val_labels=None):
        self.val_loss = val_loss
        self.val_outputs = val_outputs
        self.val_labels = val_labels

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

    scaler.scale.assert_called_with(loss)
    scaler.step.assert_called_with(optimizer)
    scaler.update.assert_called()

# --------------------------
# Training / Validation Tests
# --------------------------
def test_train_one_epoch_returns_loss(dummy_model, small_loader):
    head = DummyHead(small_loader)
    optimizer = optim.Adam(dummy_model.parameters())
    state = DummyState(dummy_model, optimizer)
    epoch_res = DummyEpochResult()

    # Patch unwrap_model to return the model output directly for dummy model
    with patch("train_model.train_loops.unwrap_model", side_effect=lambda m, x, task_name=None: m(x)):
        loss = train_loops.train_one_epoch(head, state, epoch_res)

    assert loss >= 0
    assert hasattr(epoch_res, "train_loss")

def test_validate_returns_metrics(dummy_model, small_loader):
    head = DummyHead(small_loader)
    optimizer = optim.Adam(dummy_model.parameters())
    state = DummyState(dummy_model, optimizer)
    epoch_res = DummyEpochResult()

    # Patch unwrap_model to return the tensor directly
    with patch("train_model.train_loops.unwrap_model", side_effect=lambda m, x, task_name=None: m(x)):
        val_loss = train_loops.validate(head, state, epoch_res)

    assert val_loss >= 0
    assert hasattr(epoch_res, "val_loss")
    assert hasattr(epoch_res, "val_outputs")
    assert hasattr(epoch_res, "val_labels")

    # Check that lists exist (they might be empty if loader is empty)
    assert isinstance(epoch_res.val_outputs, list)
    assert isinstance(epoch_res.val_labels, list)

    # Only check contents if non-empty
    if len(epoch_res.val_labels) > 0:
        first_batch_labels = next(iter(small_loader))[1]
        assert torch.equal(epoch_res.val_labels[0], first_batch_labels)

def test_train_one_epoch_empty_loader(dummy_model):
    empty_loader = []
    head = DummyHead(empty_loader)
    optimizer = optim.Adam(dummy_model.parameters())
    state = DummyState(dummy_model, optimizer)
    epoch_res = DummyEpochResult()

    loss = train_loops.train_one_epoch(head, state, epoch_res)
    assert loss == 0.0

def test_validate_empty_loader(dummy_model):
    empty_loader = []
    head = DummyHead(empty_loader)
    optimizer = optim.Adam(dummy_model.parameters())
    state = DummyState(dummy_model, optimizer)
    epoch_res = DummyEpochResult()

    val_loss = train_loops.validate(head, state, epoch_res)
    assert val_loss == 0.0
