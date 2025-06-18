from torch.optim.lr_scheduler import _LRScheduler
import numpy as np


class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, max_steps, warmup_ratio, warmup_steps=None, warmup_start_lr=1e-6, eta_min=1e-10):
        self.max_steps = max_steps
        if warmup_steps is not None:
            assert warmup_steps <= max_steps
            self.warmup_steps = warmup_steps
        else:
            assert 0.0 <= warmup_ratio <= 1.0
            self.warmup_steps = int(max_steps * warmup_ratio)
        
        base_lrs = [x['lr'] for x in optimizer.param_groups]
        init_opt_lr = base_lrs[0]
        if warmup_start_lr > init_opt_lr:
            warmup_start_lr = init_opt_lr / 10
            print(f"Warning: warmup_start_lr is larger than the initial learning rate. Setting warmup_start_lr to {warmup_start_lr}")
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer)
        self.base_lrs = base_lrs
        self.last_lrs = [self.warmup_start_lr] * len(self.base_lrs)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warm-up phase
            warmup_slope = (self.base_lrs[-1] - self.warmup_start_lr) / self.warmup_steps
            lrs = [self.warmup_start_lr + warmup_slope * self.last_epoch] * len(self.base_lrs)
        else:
            # Cosine annealing phase
            cosine_decay = 0.5 * (1 + np.cos(np.pi * (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)))
            lrs = [self.eta_min + (base_lr - self.eta_min) * cosine_decay for base_lr in self.base_lrs]

        self.last_lrs = lrs
        return lrs
        
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def get_last_lr(self):
        return self.last_lrs


class DummyScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        lr_list = []
        for param_group in optimizer.param_groups:
            lr_list.append(param_group['lr'])
        # assert that all elements in the list are equal
        assert np.allclose(lr_list, lr_list[0] * np.ones(len(lr_list)))
        self.constant_lr = lr_list[0]
        
    def step(self):
        pass

    def get_last_lr(self):
        return [self.constant_lr]


if __name__ == "__main__":
    import torch
    # Define the parameters for the scheduler
    max_steps = 20000
    # warmup_epochs = 5
    warmup_ratio=0.1
    warmup_start_lr = 1e-6
    eta_min = 1e-10

    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)
    # Create the scheduler
    scheduler = WarmupCosineAnnealingLR(optimizer, max_steps, warmup_ratio, warmup_start_lr, eta_min)

    # Training loop
    used_lr = []
    last_lr_list = []
    for step in range(max_steps):
        # loss = torch.sum(model(torch.randn(10, 10)))
        # get the learning rate
        lr = optimizer.param_groups[0]['lr']
        used_lr.append(lr)
        last_lr_list.append(scheduler.get_last_lr())
        scheduler.step()
    breakpoint()
    np.save("used_lr.npy", np.array(used_lr))