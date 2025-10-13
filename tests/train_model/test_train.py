# tests/train_model/test_train.py
from unittest.mock import patch, MagicMock
import pytest
import sys
import numpy as np
from train_model.metrics.metrics import Metrics
from train_model.train_state import EpochResult
from train_model import train

# --------------------------
# Test parse_args
# --------------------------
def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py"])
    args = train.parse_args()
    
    assert args.tasks == ["identification", "modifier"]
    assert args.epochs == train.Config.EPOCHS
    assert args.log_dir == "logs"
    assert args.freeze_strategy == "none"

def test_parse_args_custom(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--tasks", "modifier", "--epochs", "10", "--log-dir", "/tmp/logs"])
    args = train.parse_args()
    
    assert args.tasks == ["modifier"]
    assert args.epochs == 10
    assert args.log_dir == "/tmp/logs"

# --------------------------
# Test main loop
# --------------------------
@patch("train_model.train.create_head")
@patch("train_model.train.prepare_training")
@patch("train_model.train.train_one_epoch")
@patch("train_model.train.validate")
@patch("train_model.train.handle_checkpoints")
@patch("train_model.train.load_checkpoint")
@patch("train_model.train.StrategyFactory.from_cli")
def test_main_multiple_heads(
    mock_strategy_factory,
    mock_load_checkpoint,
    mock_handle_checkpoints,
    mock_validate,
    mock_train,
    mock_prepare,
    mock_create_head
):
    mock_load_checkpoint.return_value = None

    # Two dummy heads
    head1 = MagicMock()
    head1.load_dataloaders = MagicMock()
    head1.compute_metrics = MagicMock()
    head1.log_metrics = MagicMock()
    head2 = MagicMock()
    head2.load_dataloaders = MagicMock()
    head2.compute_metrics = MagicMock()
    head2.log_metrics = MagicMock()
    mock_create_head.side_effect = [head1, head2]

    # Dummy TrainingState
    dummy_state = MagicMock()
    dummy_state.start_epoch = 1
    dummy_state.scheduler = MagicMock()
    dummy_state.scheduler.step = MagicMock()
    dummy_state.best_val_loss = float("inf")
    dummy_state.early_stopping.early_stop = False
    dummy_state.early_stopping.step = MagicMock()
    dummy_state.writer = MagicMock()
    dummy_state.device = "cpu"
    dummy_state.model = MagicMock()
    dummy_state.optimizer = MagicMock()
    mock_prepare.return_value = dummy_state

    # Mock train/validate to set epoch_res attributes
    def fake_train(head, state, epoch_res):
        setattr(epoch_res, "train_loss", 0.1)
        return 0.1
    def fake_validate(head, state, epoch_res):
        setattr(epoch_res, "val_loss", 0.2)
        return 0.2
    mock_train.side_effect = fake_train
    mock_validate.side_effect = fake_validate

    # Patch args
    with patch("train_model.train.parse_args") as mock_args:
        mock_args.return_value = MagicMock(
            tasks=["identification", "modifier"],
            epochs=1,
            patience=5,
            log_dir="/tmp/logs",
            freeze_strategy="none",
            checkpoint=None,
            checkpoint_interval=5,
            resume=None
        )
        mock_strategy_factory.return_value = MagicMock()
        train.main()

    # Assertions
    assert mock_create_head.call_count == 2
    assert mock_train.call_count == 2  # one per head
    assert mock_validate.call_count == 2
    mock_prepare.assert_called_once()

@patch("train_model.train.create_head")
@patch("train_model.train.prepare_training")
@patch("train_model.train.train_one_epoch")
@patch("train_model.train.validate")
@patch("train_model.train.handle_checkpoints")
@patch("train_model.train.load_checkpoint")
@patch("train_model.train.StrategyFactory.from_cli")
@patch("train_model.train.parse_args")
def test_main_resume(
    mock_args,
    mock_strategy_factory,
    mock_load_checkpoint,
    mock_handle_checkpoints,
    mock_validate,
    mock_train,
    mock_prepare,
    mock_create_head,
):
    # Dummy checkpoint
    dummy_checkpoint = {"model": MagicMock()}
    mock_load_checkpoint.return_value = dummy_checkpoint

    # Dummy head
    dummy_head = MagicMock()
    dummy_head.load_dataloaders = MagicMock()
    dummy_head.compute_metrics = MagicMock()
    dummy_head.log_metrics = MagicMock()
    mock_create_head.return_value = dummy_head

    # Dummy TrainingState
    dummy_state = MagicMock()
    dummy_state.start_epoch = 3
    dummy_state.scheduler = MagicMock()
    dummy_state.scheduler.step = MagicMock()
    dummy_state.best_val_loss = float("inf")
    dummy_state.early_stopping.early_stop = False
    dummy_state.early_stopping.step = MagicMock()
    dummy_state.writer = MagicMock()
    dummy_state.model = MagicMock()
    dummy_state.optimizer = MagicMock()
    mock_prepare.return_value = dummy_state

    # train/validate mocks
    def train_mock(head, state, epoch_res):
        epoch_res.train_loss = 0.1
        return 0.1

    def validate_mock(head, state, epoch_res):
        epoch_res.val_loss = 0.2
        return 0.2

    mock_train.side_effect = train_mock
    mock_validate.side_effect = validate_mock

    # Patch args
    mock_args.return_value = MagicMock(
        tasks=["identification"],
        epochs=5,
        patience=5,
        log_dir="/tmp/logs",
        freeze_strategy="none",
        checkpoint=None,
        checkpoint_interval=5,
        resume="best_model.pth"
    )
    mock_strategy_factory.return_value = MagicMock()

    # Run main
    import train_model.train as train
    train.main()

    # Assertions
    mock_load_checkpoint.assert_called_once_with("best_model.pth")
    mock_create_head.assert_called_once()
    mock_prepare.assert_called_once()

    # Since start_epoch=3 and epochs=5, should run 3 times for 1 head
    assert mock_train.call_count == 3
    assert mock_validate.call_count == 3


@patch("train_model.train.create_head")
@patch("train_model.train.prepare_training")
@patch("train_model.train.train_one_epoch")
@patch("train_model.train.validate")
@patch("train_model.train.handle_checkpoints")
def test_main_early_stop(mock_handle_checkpoints, mock_validate, mock_train, mock_prepare, mock_create_head):
    head = MagicMock()
    head.load_dataloaders = MagicMock()
    head.compute_metrics = MagicMock()
    head.log_metrics = MagicMock()
    mock_create_head.return_value = head

    # Dummy state with early stop immediately
    state = MagicMock()
    state.start_epoch = 1
    state.scheduler = MagicMock()
    state.scheduler.step = MagicMock()
    state.best_val_loss = float("inf")
    state.early_stopping.early_stop = True
    state.early_stopping.step = MagicMock()
    state.writer = MagicMock()
    state.model = MagicMock()
    state.optimizer = MagicMock()
    mock_prepare.return_value = state

    mock_train.side_effect = lambda h, s, e: setattr(e, "train_loss", 0.1) or 0.1
    mock_validate.side_effect = lambda h, s, e: setattr(e, "val_loss", 0.2) or 0.2

    with patch("train_model.train.parse_args") as mock_args:
        mock_args.return_value = MagicMock(
            tasks=["identification"],
            epochs=5,
            patience=5,
            log_dir="/tmp/logs",
            freeze_strategy="none",
            checkpoint=None,
            checkpoint_interval=5,
            resume=None
        )
        train.main()

    mock_train.assert_called_once()
    mock_validate.assert_called_once()
    mock_handle_checkpoints.assert_called_once()
