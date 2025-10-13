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
        self.metrics = None  # optional, used in validate

    # Dummy method so prepare_training can call it
    def init_criterion(self, model, device):
        self.criterion = torch.nn.CrossEntropyLoss()


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
    dummy_heads = [DummyHead("id", 5, DummyLoader(10)), DummyHead("mod", 3, DummyLoader(10))]
    mock_instance = MagicMock()
    mock_instance.to.return_value = mock_instance
    mock_model_class.return_value = mock_instance

    model, device = train_setup.build_model(dummy_heads)

    mock_model_class.assert_called_once_with(head_configs={"id": 5, "mod": 3})
    assert model == mock_instance


@patch("train_model.train_setup.SummaryWriter")
@patch("train_model.train_setup.apply_checkpoint")
@patch("models.models.MultiHeadModel")
def test_prepare_training_basic(mock_model_class, mock_apply_checkpoint, mock_writer):
    loader = DummyLoader(10)
    heads = [DummyHead("id", 5, loader)]

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(4, 4))
            self.heads = nn.ModuleDict({"id": nn.Linear(4, 5)})
        def forward(self, x):
            return self.heads["id"](self.backbone(x).mean(dim=1))

    tiny = TinyModel()
    mock_model_class.return_value = tiny

    mock_writer_instance = MagicMock()
    mock_writer.return_value = mock_writer_instance
    mock_apply_checkpoint.return_value = 1

    class DummyStrategy:
        def apply(self, model):
            optimizer = torch.optim.Adam(model.parameters())
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            for p in model.backbone.parameters():
                p.requires_grad = False
            return model, optimizer, scheduler

    strategy = DummyStrategy()

    state = train_setup.prepare_training(
        heads,
        strategy=strategy,
        log_dir="/tmp/logs",
        checkpoint=None
    )

    for p in tiny.backbone.parameters():
        assert p.requires_grad is False

    assert isinstance(state, train_setup.TrainingState)
    assert state.writer == mock_writer_instance
    assert state.start_epoch == 1


@patch("train_model.train_setup.apply_checkpoint")
@patch("models.models.MultiHeadModel")
def test_prepare_training_with_checkpoint(mock_model_class, mock_apply_checkpoint):
    loader = DummyLoader(10)
    heads = [DummyHead("id", 5, loader)]

    # Tiny real model for optimizer
    class TinyModel2(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(2, 2))
            self.heads = nn.ModuleDict({"id": nn.Linear(2, 5)})
    tiny = TinyModel2()
    mock_model_class.return_value = tiny

    # Dummy strategy with apply method
    class DummyStrategy:
        def apply(self, model):
            optimizer = torch.optim.Adam(model.parameters())
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return model, optimizer, scheduler

    strategy = DummyStrategy()

    mock_apply_checkpoint.return_value = 5

    state = train_setup.prepare_training(
        heads,
        strategy=strategy,
        log_dir="/tmp/logs",
        checkpoint={"dummy": 1}
    )

    mock_apply_checkpoint.assert_called_once()
    assert state.start_epoch == 5


@patch("models.models.MultiHeadModel")
def test_prepare_training_scheduler_choice(mock_model_class):
    # Small dataset case
    small_loader = DummyLoader(10)
    heads_small = [DummyHead("h", 5, small_loader)]

    class TinyModel3(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(2, 2))
            self.heads = nn.ModuleDict({"h": nn.Linear(2, 5)})
    mock_model_class.return_value = TinyModel3()

    # Dummy strategy
    class DummyStrategy:
        def apply(self, model):
            optimizer = torch.optim.Adam(model.parameters())
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return model, optimizer, scheduler

    strategy = DummyStrategy()

    state_small = train_setup.prepare_training(heads_small, strategy=strategy, log_dir="/tmp/logs")

    import torch.optim.lr_scheduler as sched
    assert isinstance(state_small.scheduler, sched.StepLR)

    # Medium dataset case
    medium_loader = DummyLoader(200)
    heads_medium = [DummyHead("h", 5, medium_loader)]
    mock_model_class.return_value = TinyModel3()
    state_medium = train_setup.prepare_training(heads_medium, strategy=strategy, log_dir="/tmp/logs")
    assert isinstance(state_medium.scheduler, torch.optim.lr_scheduler.StepLR)
