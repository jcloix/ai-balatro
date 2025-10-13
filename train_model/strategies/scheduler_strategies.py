import torch
import torch.optim.lr_scheduler as sched
from typing import Optional


class SchedulerStrategy:
    """Base class for learning rate scheduler strategies, wrapping a PyTorch scheduler."""

    def __init__(self):
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

    def wrapper_build(self, optimizer: torch.optim.Optimizer):
        """Initialize the inner scheduler (to be implemented by subclasses)."""
        raise NotImplementedError

    def step(self, epoch_val_loss: Optional[float] = None):
        """Custom step logic (called manually during training)."""
        if self.scheduler is None:
            raise RuntimeError("Scheduler not initialized. Call build_(optimizer) first.")
        # Default: step with no args
        self.scheduler.step()

    def __getattr__(self, item):
        """
        Delegate all other attribute/method access to the inner scheduler,
        so you can call state_dict(), get last_epoch, etc.
        """
        if self.scheduler is not None and hasattr(self.scheduler, item):
            return getattr(self.scheduler, item)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")


class CosineAnnealing(SchedulerStrategy):
    def __init__(self, T_max=50, eta_min=0.0):
        super().__init__()
        self.T_max = T_max
        self.eta_min = eta_min

    def wrapper_build(self, optimizer):
        self.scheduler = sched.CosineAnnealingLR(optimizer, T_max=self.T_max, eta_min=self.eta_min)
        return self.scheduler


class ReduceLROnPlateau(SchedulerStrategy):
    def __init__(self, factor=0.1, patience=5, mode="min", threshold=1e-4, verbose=False):
        super().__init__()
        self.factor = factor
        self.patience = patience
        self.mode = mode
        self.threshold = threshold
        self.verbose = verbose

    def wrapper_build(self, optimizer):
        self.scheduler = sched.ReduceLROnPlateau(
            optimizer,
            factor=self.factor,
            patience=self.patience,
            mode=self.mode,
            threshold=self.threshold,
            verbose=self.verbose,
        )
        return self.scheduler

    def step(self, epoch_val_loss=None):
        if epoch_val_loss is None:
            raise ValueError("ReduceLROnPlateau requires a validation loss to step().")
        self.scheduler.step(epoch_val_loss)


class StepLR(SchedulerStrategy):
    def __init__(self, step_size=5, gamma=0.5):
        super().__init__()
        self.step_size = step_size
        self.gamma = gamma

    def wrapper_build(self, optimizer):
        self.scheduler = sched.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        return self.scheduler


class NoScheduler(SchedulerStrategy):
    def wrapper_build(self, optimizer):
        self.scheduler = None
        return None

    def step(self, epoch_val_loss=None):
        pass

    def state_dict(self):
        # Return empty dict so saving always works
        return {}
