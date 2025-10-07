import torch.optim as optim

class OptimizerStrategy:
    """Base optimizer strategy."""
    def build(self, model):
        raise NotImplementedError


class SimpleAdamStrategy(OptimizerStrategy):
    def __init__(self, lr=1e-4):
        self.lr = lr

    def build(self, model):
        return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr)

class GroupAdamWStrategy(OptimizerStrategy):
    #optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    def __init__(self, lr_backbone=1e-5, lr_heads=1e-4, weight_decay=1e-4):
        self.lr_backbone = lr_backbone
        self.lr_heads = lr_heads
        self.weight_decay = weight_decay

    def build(self, model):
        backbone_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
        head_params = [p for n, p in model.heads.named_parameters() if p.requires_grad]

        return optim.AdamW([
            {"params": backbone_params, "lr": self.lr_backbone},
            {"params": head_params, "lr": self.lr_heads},
        ], weight_decay=self.weight_decay)


