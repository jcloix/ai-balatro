import sys
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader, TensorDataset
from train_model import train
from train_model.metrics import Metrics

# -----------------------------
# 1. Default argument parsing
# -----------------------------
def test_parse_args_defaults():
    test_argv = ["train.py"]
    with patch("sys.argv", test_argv):
        args = train.parse_args()
        from train_model.train_config import Config
        assert args.epochs == Config.EPOCHS
        assert args.batch_size == Config.BATCH_SIZE
        assert args.lr == Config.LEARNING_RATE
        assert args.val_split == Config.VAL_SPLIT
        assert args.log_dir == "logs"

# -----------------------------
# 2. Argument overrides
# -----------------------------
def test_parse_args_overrides():
    test_argv = [
        "train.py",
        "--epochs", "10",
        "--batch-size", "8",
        "--lr", "0.01",
        "--val-split", "0.2",
        "--log-dir", "my_logs",
        "--num-classes", "5",
        "--use-augmented",
        "--use-weighted-sampler"
    ]
    with patch("sys.argv", test_argv):
        args = train.parse_args()
        assert args.epochs == 10
        assert args.batch_size == 8
        assert args.lr == 0.01
        assert args.val_split == 0.2
        assert args.log_dir == "my_logs"
        assert args.num_classes == 5
        assert args.no_augmented is False
        assert args.use_weighted_sampler is True

def test_parse_args_defaults_full():
    test_argv = ["train.py"]
    with patch.object(sys, "argv", test_argv):
        args = train.parse_args()

    expected_args = {
        "epochs",
        "batch_size",
        "lr",
        "val_split",
        "log_dir",
        "checkpoint_interval",
        "num_classes",
        "no_augmented",
        "freeze_backbone",
        "resume",
        "use_weighted_sampler",
        "train_transform",
        "val_transform"
    }

    # Convert Namespace to set of attribute names
    args_keys = set(vars(args).keys())

    # This will fail if a new argument is added but not in expected_args
    assert args_keys == expected_args

# -----------------------------
# Main loop tests
# -----------------------------

# -----------------------------
# 1. Helper to create dummy DataLoader
# -----------------------------
@patch("train_model.train.prepare_training")
@patch("train_model.train.load_dataloaders")
@patch("train_model.train.train_one_epoch")
@patch("train_model.train.validate")
@patch("train_model.train.log_epoch_stats")
@patch("train_model.train.handle_checkpoints")
@patch("train_model.train.parse_args")
def test_main_loop(
    mock_parse_args,
    mock_handle_ckpt,
    mock_log_stats,
    mock_validate,
    mock_train_epoch,
    mock_loaders,
    mock_prepare
):
    # --- Mock command-line args ---
    mock_parse_args.return_value = SimpleNamespace(
        batch_size=2,
        val_split=0.1,
        no_augmented=False,
        use_weighted_sampler=False,
        epochs=2,
        lr=0.01,
        log_dir="logs",
        checkpoint_interval=5,
        num_classes=3,
        train_transform="train",
        val_transform="test"
    )

    # --- Create dummy dataset + loader ---
    x = torch.randn(4, 10)
    y = torch.randint(0, 3, (4,))
    dataset = TensorDataset(x, y)
    dataset.classes = ["class0", "class1", "class2"]  # Add .classes attribute
    dummy_loader = DataLoader(dataset, batch_size=2)
    mock_loaders.return_value = (dummy_loader, dummy_loader)

    # --- Mock training state ---
    dummy_state = MagicMock()
    dummy_state.model = MagicMock()
    dummy_state.device = "cpu"
    dummy_state.criterion = MagicMock()
    dummy_state.optimizer = MagicMock()
    dummy_state.scaler = None
    dummy_state.scheduler = MagicMock()
    dummy_state.early_stopping = MagicMock()
    dummy_state.early_stopping.early_stop = False
    dummy_state.best_val_loss = 1.0
    dummy_state.writer = MagicMock()
    mock_prepare.return_value = dummy_state

    # --- Mock train/validate/checkpoint outputs ---
    mock_train_epoch.return_value = 0.1
    mock_validate.return_value = Metrics(val_loss=0.2, topk_acc=0.9, cm=torch.zeros(3, 3))
    mock_handle_ckpt.return_value = 0.1

    # --- Run main ---
    import train_model.train as train
    train.main()

    # --- Assertions ---
    assert mock_train_epoch.call_count == 2
    assert mock_validate.call_count == 2
    assert mock_log_stats.call_count == 2
    assert dummy_state.scheduler.step.call_count == 2
    assert dummy_state.early_stopping.step.call_count == 2
    dummy_state.writer.close.assert_called_once()


@patch("train_model.train.prepare_training")
@patch("train_model.train.load_dataloaders")
@patch("train_model.train.train_one_epoch")
@patch("train_model.train.validate")
@patch("train_model.train.log_epoch_stats")
@patch("train_model.train.handle_checkpoints")
@patch("train_model.train.parse_args")
def test_main_loop_zero_epochs(
    mock_parse_args,
    mock_handle_ckpt,
    mock_log_stats,
    mock_validate,
    mock_train_epoch,
    mock_loaders,
    mock_prepare
):
    # Mock args with 0 epochs
    mock_parse_args.return_value = SimpleNamespace(
        batch_size=2,
        val_split=0.1,
        no_augmented=False,
        use_weighted_sampler=False,
        epochs=0,
        lr=0.01,
        log_dir="logs",
        checkpoint_interval=5,
        num_classes=3,
        train_transform="train",
        val_transform="test"
    )

    # Dummy loader
    x = torch.randn(2, 10)
    y = torch.randint(0, 3, (2,))
    dataset = TensorDataset(x, y)
    dataset.classes = ["class0", "class1", "class2"]
    dummy_loader = DataLoader(dataset, batch_size=2)
    mock_loaders.return_value = (dummy_loader, dummy_loader)

    # Dummy state
    dummy_state = MagicMock()
    dummy_state.writer = MagicMock()
    mock_prepare.return_value = dummy_state

    train.main()

    # Ensure train/validate/log are never called
    assert mock_train_epoch.call_count == 0
    assert mock_validate.call_count == 0
    assert mock_log_stats.call_count == 0
    dummy_state.writer.close.assert_called_once()


@patch("train_model.train.prepare_training")
@patch("train_model.train.load_dataloaders")
@patch("train_model.train.train_one_epoch")
@patch("train_model.train.validate")
@patch("train_model.train.log_epoch_stats")
@patch("train_model.train.handle_checkpoints")
@patch("train_model.train.parse_args")
def test_main_loop_early_stop_first_epoch(
    mock_parse_args,
    mock_handle_ckpt,
    mock_log_stats,
    mock_validate,
    mock_train_epoch,
    mock_loaders,
    mock_prepare
):
    mock_parse_args.return_value = SimpleNamespace(
        batch_size=2,
        val_split=0.1,
        no_augmented=False,
        use_weighted_sampler=False,
        epochs=5,
        lr=0.01,
        log_dir="logs",
        checkpoint_interval=5,
        num_classes=3,
        train_transform="train",
        val_transform="test"
    )

    # Dummy loader
    x = torch.randn(2, 10)
    y = torch.randint(0, 3, (2,))
    dataset = TensorDataset(x, y)
    dataset.classes = ["class0", "class1", "class2"]
    dummy_loader = DataLoader(dataset, batch_size=2)
    mock_loaders.return_value = (dummy_loader, dummy_loader)

    # Dummy state
    dummy_state = MagicMock()
    dummy_state.model = MagicMock()
    dummy_state.device = "cpu"
    dummy_state.criterion = MagicMock()
    dummy_state.optimizer = MagicMock()
    dummy_state.scaler = None
    dummy_state.scheduler = MagicMock()
    dummy_state.early_stopping = MagicMock()
    dummy_state.early_stopping.early_stop = False
    dummy_state.best_val_loss = 1.0
    dummy_state.writer = MagicMock()
    mock_prepare.return_value = dummy_state

    # Mock functions
    mock_train_epoch.return_value = 0.1
    mock_validate.return_value = Metrics(val_loss=0.2, topk_acc=0.9, cm=torch.zeros(3, 3))
    mock_handle_ckpt.return_value = 0.1

    # Trigger early stopping after first epoch
    def early_stop_side_effect(val_loss):
        dummy_state.early_stopping.early_stop = True
    dummy_state.early_stopping.step.side_effect = early_stop_side_effect

    train.main()

    # Loop should run only once
    assert mock_train_epoch.call_count == 1
    assert mock_validate.call_count == 1
    assert mock_log_stats.call_count == 1
    dummy_state.writer.close.assert_called_once()

@patch("train_model.train.prepare_training")
@patch("train_model.train.load_dataloaders")
@patch("train_model.train.train_one_epoch")
@patch("train_model.train.validate")
@patch("train_model.train.log_epoch_stats")
@patch("train_model.train.handle_checkpoints")
@patch("train_model.train.parse_args")
def test_main_loop_empty_dataset(
    mock_parse_args,
    mock_handle_ckpt,
    mock_log_stats,
    mock_validate,
    mock_train_epoch,
    mock_loaders,
    mock_prepare
):
    mock_parse_args.return_value = SimpleNamespace(
        batch_size=2,
        val_split=0.1,
        no_augmented=False,
        use_weighted_sampler=False,
        epochs=2,
        lr=0.01,
        log_dir="logs",
        checkpoint_interval=5,
        num_classes=3,
        train_transform="train",
        val_transform="test"
    )

    # Empty DataLoader
    class EmptyDataset:
        classes = ["class0", "class1", "class2"]
        def __len__(self): return 0
    empty_loader = DataLoader(EmptyDataset())
    mock_loaders.return_value = (empty_loader, empty_loader)

    dummy_state = MagicMock()
    dummy_state.model = MagicMock()
    dummy_state.device = "cpu"
    dummy_state.criterion = MagicMock()
    dummy_state.optimizer = MagicMock()
    dummy_state.scaler = None
    dummy_state.scheduler = MagicMock()
    dummy_state.early_stopping = MagicMock()
    dummy_state.early_stopping.early_stop = False
    dummy_state.best_val_loss = 1.0
    dummy_state.writer = MagicMock()
    mock_prepare.return_value = dummy_state

    mock_train_epoch.return_value = 0.0
    mock_validate.return_value = Metrics(val_loss=0.0, topk_acc=0.0, cm=torch.zeros(3,3))
    mock_handle_ckpt.return_value = 0.0

    train.main()

    assert mock_train_epoch.call_count == 2
    assert mock_validate.call_count == 2
    dummy_state.writer.close.assert_called_once()

