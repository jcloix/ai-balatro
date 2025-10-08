import torch.optim as optim
from typing import Optional

class OptimizerStrategy:
    """Base optimizer strategy that wraps a PyTorch optimizer."""

    def __init__(self):
        self.optimizer: Optional[optim.Optimizer] = None

    def wrapper_build(self, model):
        """Build and store the underlying optimizer."""
        raise NotImplementedError

    def __getattr__(self, item):
        """Delegate all other attributes/methods to the underlying optimizer."""
        if self.optimizer is not None and hasattr(self.optimizer, item):
            return getattr(self.optimizer, item)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")


class SimpleAdamStrategy(OptimizerStrategy):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.lr = lr

    def wrapper_build(self, model):
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr
        )
        return self.optimizer


class GroupAdamWStrategy(OptimizerStrategy):
    def __init__(self, lr_backbone=1e-5, lr_heads=1e-4, weight_decay=1e-4):
        super().__init__()
        self.lr_backbone = lr_backbone
        self.lr_heads = lr_heads
        self.weight_decay = weight_decay

    def wrapper_build(self, model):
        backbone_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
        head_params = [p for n, p in model.heads.named_parameters() if p.requires_grad]

        self.optimizer = optim.AdamW([
            {"params": backbone_params, "lr": self.lr_backbone},
            {"params": head_params, "lr": self.lr_heads},
        ], weight_decay=self.weight_decay)
        return self.optimizer
