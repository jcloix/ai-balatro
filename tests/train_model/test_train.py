import torch
from torch import nn
from unittest.mock import MagicMock, patch
import pytest
import sys
from torch.utils.data import DataLoader, TensorDataset
from types import SimpleNamespace

# Modules to test
from train_model import train

@pytest.fixture
def dummy_model():
    model = nn.Linear(10, 2)
    return model

@pytest.fixture
def dummy_loader():
    # Small dataset: 3 batches of 2 samples
    X = [torch.randn(2, 10) for _ in range(3)]
    y = [torch.randint(0, 2, (2,)) for _ in range(3)]
    return list(zip(X, y))

def test_train_validate_epoch():
    # Dummy dataset
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))
    dataset = TensorDataset(x, y)
    dummy_loader = DataLoader(dataset, batch_size=2)

    # Dummy model
    dummy_model = torch.nn.Linear(10, 2)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.01)
    device = torch.device("cpu")

    # Train one epoch
    train_loss = train.train_one_epoch(dummy_model, dummy_loader, criterion, optimizer, device)
    assert train_loss >= 0

    # Validate
    val_loss = train.validate(dummy_model, dummy_loader, criterion, device)
    assert val_loss >= 0

def test_forward_backward(dummy_model):
    x = torch.randn(2, 10)
    y = torch.randint(0, 2, (2,))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(dummy_model.parameters())

    # No scaler
    out, loss = train.forward_pass(dummy_model, x, y, criterion)
    train.backward_step(loss, optimizer)
    assert loss.item() >= 0

    # With dummy scaler
    scaler = MagicMock()
    out, loss = train.forward_pass(dummy_model, x, y, criterion, scaler=scaler)
    train.backward_step(loss, optimizer, scaler=scaler)

@patch("train_model.train.load_merged_labels")
@patch("train_model.train.get_train_val_loaders")
def test_load_dataloaders(mock_get_loaders, mock_load_labels):
    mock_load_labels.return_value = {"img1.png": {"label": 0}, "img2.png": {"label": 1}}
    mock_get_loaders.return_value = ("train_loader", "val_loader")

    train_loader, val_loader = train.load_dataloaders(batch_size=2, val_split=0.5)
    assert train_loader == "train_loader"
    assert val_loader == "val_loader"


def test_parse_args_defaults():
    test_argv = ["train.py"]
    with patch.object(sys, "argv", test_argv):
        args = train.parse_args()
        from train_model.train_config import Config
        assert args.epochs == Config.EPOCHS
        assert args.batch_size == Config.BATCH_SIZE
        assert args.lr == Config.LEARNING_RATE
        assert args.val_split == Config.VAL_SPLIT
        assert args.log_dir == "logs"

def test_parse_args_overrides():
    test_argv = [
        "train.py",
        "--epochs", "10",
        "--batch-size", "8",
        "--lr", "0.01",
        "--val-split", "0.2",
        "--log-dir", "my_logs",
        "--num-classes", "5",
        "--use-augmented"
    ]
    with patch.object(sys, "argv", test_argv):
        args = train.parse_args()
        assert args.epochs == 10
        assert args.batch_size == 8
        assert args.lr == 0.01
        assert args.val_split == 0.2
        assert args.log_dir == "my_logs"
        assert args.num_classes == 5
        assert args.use_augmented is True

# -----------------------------
# 2. Test handle_checkpoints
# -----------------------------
@patch("train_model.train.torch.save")
@patch("train_model.train.os.makedirs")
def test_handle_checkpoints_saves(mock_makedirs, mock_torch_save):
    class DummyModel:
        def state_dict(self):
            return {"dummy": 123}

    model = DummyModel()
    best_val_loss = 1.0
    val_loss = 0.5
    checkpoint_dir = "/fake/dir"

    # Best model save
    new_best = train.handle_checkpoints(model, val_loss, best_val_loss, 1, checkpoint_dir, checkpoint_interval=5)
    assert new_best == val_loss
    mock_torch_save.assert_called()
    mock_makedirs.assert_called()

    # Periodic checkpoint
    mock_torch_save.reset_mock()
    train.handle_checkpoints(model, val_loss=0.6, best_val_loss=new_best, epoch=5, checkpoint_dir=checkpoint_dir, checkpoint_interval=5)
    mock_torch_save.assert_called()

# -----------------------------
# 3. Test log_epoch_stats
# -----------------------------
def test_log_epoch_stats_prints(capsys):
    class DummyOptimizer:
        param_groups = [{"lr": 0.01}]

    train.log_epoch_stats(1, 0.1, 0.2, DummyOptimizer())
    captured = capsys.readouterr()
    assert "Epoch 1" in captured.out
    assert "Train Loss: 0.1000" in captured.out
    assert "Val Loss: 0.2000" in captured.out

@patch("train_model.train.SummaryWriter")
def test_log_epoch_stats_tensorboard(mock_writer_class):
    mock_writer = mock_writer_class.return_value
    class DummyOptimizer:
        param_groups = [{"lr": 0.01}]
    train.log_epoch_stats(1, 0.1, 0.2, DummyOptimizer(), writer=mock_writer)
    mock_writer.add_scalar.assert_any_call("Loss/Train", 0.1, 1)
    mock_writer.add_scalar.assert_any_call("Loss/Val", 0.2, 1)
    mock_writer.add_scalar.assert_any_call("Learning_Rate", 0.01, 1)

# -----------------------------
# 4. Test forward/backward passes
# -----------------------------
def test_forward_backward_no_scaler():
    import torch
    from torch import nn, optim

    model = nn.Linear(2,2)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    images = torch.tensor([[1.,2.],[3.,4.]])
    labels = torch.tensor([[0.,1.],[1.,0.]])

    # Forward pass without scaler
    outputs, loss = train.forward_pass(model, images, labels, criterion, scaler=None)
    assert isinstance(outputs, torch.Tensor)
    assert loss.item() > 0

    # Backward step
    train.backward_step(loss, optimizer, scaler=None)
    # Optimizer should have non-zero gradients applied
    for p in model.parameters():
        assert p.grad is not None

# -----------------------------
# 5. Test load_dataloaders calls get_train_val_loaders correctly
# -----------------------------
@patch("train_model.train.get_train_val_loaders")
@patch("train_model.train.load_merged_labels")
@patch("train_model.train.CardDataset.from_labels_dict")
def test_load_dataloaders(mock_dataset, mock_load_labels, mock_get_loaders):
    dummy_dataset = MagicMock()
    mock_dataset.return_value = dummy_dataset
    dummy_train_loader = dummy_val_loader = "loader"
    mock_get_loaders.return_value = (dummy_train_loader, dummy_val_loader)
    mock_load_labels.return_value = {"img1.png":{"label":0}}

    train_loader, val_loader = train.load_dataloaders(batch_size=2, val_split=0.1, use_augmented=False)
    assert train_loader == dummy_train_loader
    assert val_loader == dummy_val_loader
    mock_get_loaders.assert_called_once_with(
        dummy_dataset,
        batch_size=2,
        val_split=0.1,
        train_transform=train.Config.TRANSFORMS['train'],
        val_transform=train.Config.TRANSFORMS['test'],
        shuffle=True
    )

# -----------------------------
# 6. Main loop (mock full training)
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
    # Mock args with SimpleNamespace for real attributes
    mock_parse_args.return_value = SimpleNamespace(
        batch_size=2,
        val_split=0.1,
        use_augmented=False,
        epochs=2,
        lr=0.01,
        log_dir="logs",
        checkpoint_interval=5,
        num_classes=3,
        transforms="train"
    )

    # Mock loaders
    dummy_loader = ["batch1", "batch2"]
    mock_loaders.return_value = (dummy_loader, dummy_loader)

    # Mock training state
    dummy_state = MagicMock()
    dummy_state.model = MagicMock()
    dummy_state.device = "cpu"
    dummy_state.criterion = MagicMock()
    dummy_state.optimizer = MagicMock()
    dummy_state.scaler = None
    dummy_state.scheduler = MagicMock()
    dummy_state.early_stopping = MagicMock()
    dummy_state.early_stopping.early_stop = False  # <-- prevent early stop
    dummy_state.best_val_loss = 1.0
    dummy_state.writer = MagicMock()
    mock_prepare.return_value = dummy_state

    # Mock training/validation/checkpoint functions
    mock_train_epoch.return_value = 0.1
    mock_validate.return_value = 0.2
    mock_handle_ckpt.return_value = 0.1

    # Run main
    train.main()

    # Ensure main steps called correct number of times
    assert mock_train_epoch.call_count == 2
    assert mock_validate.call_count == 2
    assert mock_log_stats.call_count == 2
    assert dummy_state.scheduler.step.call_count == 2
    assert dummy_state.early_stopping.step.call_count == 2
    dummy_state.writer.close.assert_called_once()

