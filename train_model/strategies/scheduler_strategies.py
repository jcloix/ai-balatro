import torch.optim.lr_scheduler as sched
import torch

class SchedulerStrategy:
    def build(self, optimizer):
        raise NotImplementedError


class CosineAnnealing(SchedulerStrategy):
    def __init__(self, T_max=50):
        self.T_max = T_max

    def build(self, optimizer):
        return sched.CosineAnnealingLR(optimizer, T_max=self.T_max)


class ReduceLROnPlateau(SchedulerStrategy):
    def __init__(self, factor=0.1, patience=5):
        self.factor = factor
        self.patience = patience

    def build(self, optimizer):
        return sched.ReduceLROnPlateau(optimizer, factor=self.factor, patience=self.patience)
    
class StepLR(SchedulerStrategy):
    def __init__(self, step_size=5, gamma=0.5):
        self.step_size = step_size
        self.gamma = gamma

    def build(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)


class NoScheduler(SchedulerStrategy):
    def build(self, optimizer):
        return None
