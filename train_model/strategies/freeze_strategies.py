import torch.nn as nn

class FreezeStrategy:
    """Base class for layer freezing strategies."""
    def apply(self, model: nn.Module):
        raise NotImplementedError
    
    def unfreeze_partial(self, model, layers):
        for p in model.backbone.parameters():
            p.requires_grad = False
        for name, module in model.backbone.named_children():
            if name in layers:
                for p in module.parameters():
                    p.requires_grad = True
        return model


class FreezeAll(FreezeStrategy):
    def apply(self, model):
        for p in model.backbone.parameters():
            p.requires_grad = False
        return model


class UnfreezeHighLevel(FreezeStrategy):
    """Unfreeze only the last 2 layers (ResNet-style)."""
    def __init__(self, layers=("layer3", "layer4")):
        self.layers = layers

    def apply(self, model):
        return self.unfreeze_partial(model,self.layers)
    
class UnfreezeMidLevel(FreezeStrategy):
    """Unfreeze the last 3 layers (ResNet-style)."""
    def __init__(self, layers=("layer2","layer3", "layer4")):
        self.layers = layers

    def apply(self, model):
        return self.unfreeze_partial(model,self.layers)
    


class UnfreezeAll(FreezeStrategy):
    def apply(self, model):
        for p in model.backbone.parameters():
            p.requires_grad = True
        return model
