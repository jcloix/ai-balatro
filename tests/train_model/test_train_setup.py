# tests/train_model/test_train_setup.py
import pytest
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn

from train_model import train_setup
from train_model.train_state import TrainingState

# Helper dummy classes for test heads/loaders
class DummyHead:
    def __init__(self, name, num_classes, train_loader):
        self.name = name
        self.num_classes = num_classes
        self.train_loader = train_loader

class DummyLoader:
    def __init__(self, dataset_len):
        self.dataset = MagicMock()
        # ensure len(dataset) works
        self.dataset.__len__.return_value = dataset_len

# --------------------------
# Tests
# --------------------------
@patch("models.models.MultiHeadModel")
def test_build_model(mock_model_class):
    # This test only checks build_model constructs head_configs and calls MultiHeadModel.
    dummy_heads = [DummyHead("id", 5, DummyLoader(10)), DummyHead("mod", 3, DummyLoader(10))]
    mock_instance = MagicMock()
    # Simulate .to() returning the instance (as real model.to() would)
    mock_instance.to.return_value = mock_instance
    mock_model_class.return_value = mock_instance

    model, device = train_setup.build_model(dummy_heads)

    mock_model_class.assert_called_once_with(head_configs={"id": 5, "mod": 3})
    assert model == mock_instance


@patch("train_model.train_setup.SummaryWriter")
@patch("train_model.train_setup.apply_checkpoint")
@patch("models.models.MultiHeadModel")
def test_prepare_training_basic(mock_model_class, mock_apply_checkpoint, mock_writer):
    """
    Ensure prepare_training builds a model, optionally freezes backbone,
    constructs optimizer/scheduler/scaler/writer and returns a TrainingState.
    Provide a dummy real nn.Module so optimizer gets real parameters.
    """
    loader = DummyLoader(10)
    heads = [DummyHead("id", 5, loader)]

    # Create a tiny real model class (real nn.Parameters for optimizer)
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            # backbone and a head so freezing logic has something to work with
            self.backbone = nn.Sequential(nn.Linear(4, 4))
            self.heads = nn.ModuleDict({"id": nn.Linear(4, 5)})
        def forward(self, x):
            return self.heads["id"](self.backbone(x).mean(dim=1))

    tiny = TinyModel()
    # Make sure calling .to(device) returns the model (consistent with build_model)
    mock_model_class.return_value = tiny

    # Mock SummaryWriter & apply_checkpoint
    mock_writer_instance = MagicMock()
    mock_writer.return_value = mock_writer_instance
    mock_apply_checkpoint.return_value = 1  # start_epoch returned

    state = train_setup.prepare_training(
        heads,
        log_dir="/tmp/logs",
        lr=0.001,
        freeze_backbone=True,
        checkpoint=None
    )

    # Freeze check: backbone parameters must have requires_grad False
    for p in tiny.backbone.parameters():
        assert p.requires_grad is False

    # Other assertions
    assert isinstance(state, TrainingState)
    assert state.writer == mock_writer_instance
    assert state.start_epoch == 1


@patch("train_model.train_setup.apply_checkpoint")
@patch("models.models.MultiHeadModel")
def test_prepare_training_with_checkpoint(mock_model_class, mock_apply_checkpoint):
    """
    When a checkpoint is passed, prepare_training should call apply_checkpoint and
    return TrainingState with start_epoch as returned by apply_checkpoint.
    """
    loader = DummyLoader(10)
    heads = [DummyHead("id", 5, loader)]

    # Provide a tiny real model (parameters present)
    class TinyModel2(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(2, 2))
            self.heads = nn.ModuleDict({"id": nn.Linear(2, 5)})
    tiny = TinyModel2()
    mock_model_class.return_value = tiny

    # Mock apply_checkpoint returning a specific epoch
    mock_apply_checkpoint.return_value = 5

    state = train_setup.prepare_training(heads, log_dir="/tmp/logs", checkpoint={"dummy": 1})

    mock_apply_checkpoint.assert_called_once()
    assert state.start_epoch == 5


@patch("models.models.MultiHeadModel")
def test_prepare_training_scheduler_choice(mock_model_class):
    """
    Ensure scheduler selection works based on dataset size across heads.
    - small dataset -> StepLR
    - medium dataset (>=100) -> ReduceLROnPlateau
    """
    # Small dataset case
    small_loader = DummyLoader(10)
    heads_small = [DummyHead("h", 5, small_loader)]
    # Tiny model with parameters
    class TinyModel3(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(2, 2))
            self.heads = nn.ModuleDict({"h": nn.Linear(2, 5)})
    mock_model_class.return_value = TinyModel3()
    state_small = train_setup.prepare_training(heads_small, log_dir="/tmp/logs")
    import torch.optim.lr_scheduler as sched
    assert isinstance(state_small.scheduler, sched.StepLR)

    # Medium dataset case
    medium_loader = DummyLoader(200)
    heads_medium = [DummyHead("h", 5, medium_loader)]
    mock_model_class.return_value = TinyModel3()
    state_medium = train_setup.prepare_training(heads_medium, log_dir="/tmp/logs")
    assert isinstance(state_medium.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
