# tests/train_model/test_train.py
from unittest.mock import patch, MagicMock
from train_model.metrics_v2 import Metrics
from train_model import train
import sys

# --------------------------
# Test parse_args
# --------------------------
def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py"])
    args = train.parse_args()
    
    assert args.tasks == ["identification", "modifier"]
    assert args.epochs == train.Config.EPOCHS
    assert args.log_dir == "logs"
    assert args.freeze_backbone is False

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
@patch("train_model.train.log_epoch_stats")
@patch("train_model.train.handle_checkpoints")
@patch("train_model.train.load_checkpoint")
def test_main_run(
    mock_load_checkpoint,
    mock_handle_checkpoints,
    mock_log,
    mock_validate,
    mock_train,
    mock_prepare,
    mock_create_head
):
    # Mock checkpoint load
    mock_load_checkpoint.return_value = None

    # Dummy head returned by create_head
    dummy_head = MagicMock()
    dummy_head.train_loader = MagicMock()
    dummy_head.val_loader = MagicMock()
    dummy_head.criterion = MagicMock()
    dummy_head.num_classes = 5
    dummy_head.train_loader.classes = ["A", "B"]
    mock_create_head.return_value = dummy_head

    # Dummy TrainingState returned by prepare_training
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

    # Mock training/validation outputs
    mock_train.return_value = 0.1

    mock_val_metrics = MagicMock()
    mock_val_metrics.val_loss = 0.2
    mock_val_metrics.topk_acc = 0.9
    mock_val_metrics.cm = [[1, 0], [0, 1]]
    mock_validate.return_value = mock_val_metrics

    # Patch argparse to control args
    with patch("train_model.train.parse_args") as mock_args:
        mock_args.return_value = MagicMock(
            tasks=["identification"],
            epochs=1,
            lr=None,
            patience=5,
            log_dir="/tmp/logs",
            freeze_backbone=False,
            checkpoint=None,
            checkpoint_interval=5,
            use_weighted_sampler=False,
            resume=None
        )

        # Run main (should exit normally)
        train.main()

    # Ensure mocks were called correctly
    mock_create_head.assert_called()
    mock_prepare.assert_called_once_with(
        [dummy_head],
        "/tmp/logs",
        None,
        5,
        False,
        None
    )
    mock_train.assert_called_once()
    mock_validate.assert_called_once()
    mock_log.assert_called_once()
    mock_handle_checkpoints.assert_called_once()


@patch("train_model.train.create_head")
@patch("train_model.train.prepare_training")
@patch("train_model.train.train_one_epoch")
@patch("train_model.train.validate")
@patch("train_model.train.log_epoch_stats")
@patch("train_model.train.handle_checkpoints")
@patch("train_model.train.load_checkpoint")
def test_main_multiple_heads(
    mock_load_checkpoint,
    mock_handle_checkpoints,
    mock_log,
    mock_validate,
    mock_train,
    mock_prepare,
    mock_create_head
):
    mock_load_checkpoint.return_value = None

    # Two dummy heads
    head1 = MagicMock()
    head1.train_loader = MagicMock()
    head1.val_loader = MagicMock()
    head1.criterion = MagicMock()
    head1.num_classes = 5
    head1.train_loader.classes = ["A", "B"]

    head2 = MagicMock()
    head2.train_loader = MagicMock()
    head2.val_loader = MagicMock()
    head2.criterion = MagicMock()
    head2.num_classes = 3
    head2.train_loader.classes = ["X", "Y"]

    mock_create_head.side_effect = [head1, head2]

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

    # Train/validation mocks
    mock_train.return_value = 0.1
    mock_val_metrics = MagicMock()
    mock_val_metrics.val_loss = 0.2
    mock_val_metrics.topk_acc = 0.9
    mock_val_metrics.cm = [[1]]
    mock_validate.return_value = mock_val_metrics

    with patch("train_model.train.parse_args") as mock_args:
        mock_args.return_value = MagicMock(
            tasks=["identification", "modifier"],
            epochs=1,
            lr=None,
            patience=5,
            log_dir="/tmp/logs",
            freeze_backbone=False,
            checkpoint=None,
            checkpoint_interval=5,
            use_weighted_sampler=False,
            resume=None
        )
        train.main()

    # Ensure heads were created
    assert mock_create_head.call_count == 2
    # Ensure training and validation called for both
    assert mock_train.call_count == 2
    assert mock_validate.call_count == 2
    # Logging called twice (once per head)
    assert mock_log.call_count == 2

@patch("train_model.train.parse_args")
@patch("train_model.train.handle_checkpoints")
@patch("train_model.train.log_epoch_stats")
@patch("train_model.train.Metrics.from_epoch")
@patch("train_model.train.validate")
@patch("train_model.train.train_one_epoch")
@patch("train_model.train.prepare_training")
@patch("train_model.train.create_head")
@patch("train_model.train.load_checkpoint")
def test_main_resume(
    mock_load_checkpoint,
    mock_create_head,
    mock_prepare,
    mock_train,
    mock_validate,
    mock_metrics_cls,
    mock_log,
    mock_handle_checkpoints,
    mock_args
):
    # Provide dummy checkpoint
    dummy_checkpoint = {"model": MagicMock()}
    mock_load_checkpoint.return_value = dummy_checkpoint

    # Dummy head
    dummy_head = MagicMock()
    dummy_head.train_loader = MagicMock()
    dummy_head.val_loader = MagicMock()
    dummy_head.criterion = MagicMock()
    dummy_head.num_classes = 5
    dummy_head.train_loader.classes = ["A", "B"]
    mock_create_head.return_value = dummy_head

    # Dummy TrainingState
    dummy_state = MagicMock()
    dummy_state.start_epoch = 3
    dummy_state.scheduler = MagicMock()
    dummy_state.best_val_loss = float("inf")
    dummy_state.early_stopping.early_stop = False
    dummy_state.writer = MagicMock()
    dummy_state.model = MagicMock()
    dummy_state.optimizer = MagicMock()
    dummy_state.early_stopping.step = MagicMock()
    mock_prepare.return_value = dummy_state

    # Training/validation outputs with real floats
    mock_train.return_value = 0.1
    val_metrics = MagicMock()
    val_metrics.val_loss = 0.2
    val_metrics.topk_acc = 0.9
    val_metrics.cm = [[1]]
    mock_validate.return_value = val_metrics

    # Metrics.from_epoch returns real Metrics object
    from train_model.metrics_v2 import Metrics
    mock_metrics_cls.side_effect = lambda train_loss, val_metrics: Metrics(
        train_loss=train_loss,
        val_loss=val_metrics.val_loss,
        topk_acc=val_metrics.topk_acc,
        cm=val_metrics.cm,
    )

    # Patch args
    mock_args.return_value = MagicMock(
        tasks=["identification"],
        epochs=5,
        lr=None,
        patience=5,
        log_dir="/tmp/logs",
        freeze_backbone=False,
        checkpoint=None,
        checkpoint_interval=5,
        use_weighted_sampler=False,
        resume="best_model.pth"
    )

    # Run main; should not raise any exceptions
    train.main()

    # Check calls
    mock_load_checkpoint.assert_called_once_with("best_model.pth")
    mock_create_head.assert_called_once()
    mock_prepare.assert_called_once()

    # train_one_epoch should be called once per epoch per head
    num_epochs = mock_args.return_value.epochs - dummy_state.start_epoch + 1
    num_heads = len(mock_args.return_value.tasks)
    expected_calls = num_epochs * num_heads
    assert mock_train.call_count == expected_calls


@patch("train_model.train.handle_checkpoints")
@patch("train_model.train.create_head")
@patch("train_model.train.prepare_training")
@patch("train_model.train.train_one_epoch")
@patch("train_model.train.validate")
def test_main_early_stop(mock_validate, mock_train, mock_prepare, mock_create_head, mock_handle_checkpoints):
    import numpy as np
    from train_model.metrics_v2 import Metrics

    # Dummy head
    head = MagicMock()
    head.train_loader = MagicMock()
    head.val_loader = MagicMock()
    head.criterion = MagicMock()
    head.num_classes = 2
    head.train_loader.classes = ["A", "B"]
    mock_create_head.return_value = head

    # Dummy TrainingState with early stopping triggered immediately
    state = MagicMock()
    state.start_epoch = 1
    state.scheduler = MagicMock()
    state.best_val_loss = float("inf")
    state.early_stopping.early_stop = True  # triggers stop
    state.writer = MagicMock()
    state.device = "cpu"
    state.model = MagicMock()
    state.optimizer = MagicMock()
    state.optimizer.param_groups = [{"lr": 0.001}]  # real float for formatting
    state.early_stopping.step = MagicMock()
    mock_prepare.return_value = state

    # Training/validation outputs with real floats and np.array for cm
    mock_train.return_value = 0.1
    val_metrics = Metrics(
        train_loss=0.1,
        val_loss=0.2,
        topk_acc=0.9,
        cm=np.array([[1, 0], [0, 1]])  # fix: np.array
    )
    mock_validate.return_value = val_metrics

    # Mock handle_checkpoints to return best_val_loss
    mock_handle_checkpoints.side_effect = lambda *args, **kwargs: 0.2

    # Patch args
    with patch("train_model.train.parse_args") as mock_args:
        mock_args.return_value = MagicMock(
            tasks=["identification"],
            epochs=5,
            lr=None,
            patience=5,
            log_dir="/tmp/logs",
            freeze_backbone=False,
            checkpoint=None,
            checkpoint_interval=5,
            use_weighted_sampler=False,
            resume=None
        )

        # Run main; should exit immediately due to early stop
        import train_model.train as train
        train.main()

    # Assertions
    mock_train.assert_called_once()  # early stop triggers after 1 epoch
    mock_validate.assert_called_once()
    mock_handle_checkpoints.assert_called_once()
    assert state.best_val_loss == 0.2