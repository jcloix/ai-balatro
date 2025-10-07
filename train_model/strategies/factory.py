from train_model.strategies.freeze_strategies import FreezeAll, UnfreezeHighLevel, UnfreezeMidLevel, UnfreezeAll
from train_model.strategies.optimizer_strategies import SimpleAdamStrategy, GroupAdamWStrategy
from train_model.strategies.scheduler_strategies import CosineAnnealing, ReduceLROnPlateau, StepLR, NoScheduler

class StrategyFactory:
    DEFAULTS = {
        "freeze": "none",
        "optimizer": "simple",
        "scheduler": "cosine"
    }

    @classmethod
    def from_cli(cls, args):
        freeze = args.freeze_strategy or cls.DEFAULTS["freeze"]
        optimizer = args.optimizer or cls.DEFAULTS["optimizer"]
        scheduler = args.scheduler or cls.DEFAULTS["scheduler"]
        return cls(freeze, optimizer, scheduler, args)

    def __init__(self, freeze, optimizer, scheduler, args):
        self.freeze_name = freeze
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.args = args

    # -----------------------------
    # Freeze strategy
    # -----------------------------
    def build_freeze(self):
        mapping = {
            "none": FreezeAll(),
            "high": UnfreezeHighLevel(layers=self.args.freeze_layers_high),
            "mid": UnfreezeMidLevel(layers=self.args.freeze_layers_mid),
            "all": UnfreezeAll(),
        }
        return mapping.get(self.freeze_name, UnfreezeAll())

    # -----------------------------
    # Optimizer strategy
    # -----------------------------
    def build_optimizer(self):
        mapping = {
            "simple": SimpleAdamStrategy(lr=self.args.optimizer_lr),
            "group": GroupAdamWStrategy(
                lr_backbone=self.args.optimizer_lr_backbone,
                lr_heads=self.args.optimizer_lr_heads,
                weight_decay=self.args.optimizer_weight_decay,
            ),
        }
        return mapping.get(self.optimizer_name, SimpleAdamStrategy(lr=self.args.optimizer_lr))

    # -----------------------------
    # Scheduler strategy
    # -----------------------------
    def build_scheduler(self):
        mapping = {
            "cosine": CosineAnnealing(T_max=self.args.scheduler_tmax),
            "plateau": ReduceLROnPlateau(
                factor=self.args.scheduler_factor,
                patience=self.args.scheduler_patience
            ),
            "step": StepLR(
                step_size=self.args.scheduler_step_size,
                gamma=self.args.scheduler_gamma
            ),
            "none": NoScheduler(),
        }
        return mapping.get(self.scheduler_name, CosineAnnealing(T_max=self.args.scheduler_tmax))

    # -----------------------------
    # Apply all strategies
    # -----------------------------
    def apply(self, model):
        freeze_strategy = self.build_freeze()
        freeze_strategy.apply(model)
        optimizer = self.build_optimizer().build(model)
        scheduler = self.build_scheduler().build(optimizer)
        return model, optimizer, scheduler
