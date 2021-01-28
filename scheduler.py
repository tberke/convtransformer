from typing import List

import torch
import torch.optim as optim


class InverseSquareRootLR(optim.lr_scheduler._LRScheduler):
    """Scheduler which implements the inverse square root decay scheme.

    For the first 'warmup_steps' number of steps, the learning rate
    increases linearly from 'warmup_init_lr' to 'base_lr'. Afterwards it
    decays proportionally to the inverse of the square root of the
    number of steps taken so far.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        base_lr: float,
        warmup_init_lr: float,
        warmup_steps: int,
    ) -> None:
        """Initializes a new scheduler.

        Args:
            optimizer: The associated optimizer.
            base_lr: The peak learning rate, attained after all warm-up
                steps are done.
            warmup_init_lr: The initial learning rate.
            warmup_steps: Number of warm-up steps.
        """
        self.lr = warmup_init_lr
        self.base_lr = base_lr
        self.warmup_init_lr = warmup_init_lr
        self.warmup_steps = warmup_steps
        self.warmup_step_size = (base_lr - warmup_init_lr) / warmup_steps

        super(InverseSquareRootLR, self).__init__(optimizer, last_epoch=-1)

    def get_lr(self) -> List[float]:
        """Computes the current learning rate.

        Returns:
            List containing the learning rate for each parameter group.
        """
        num_steps = max(1, self.last_epoch)

        if num_steps < self.warmup_steps:
            self.lr = self.warmup_init_lr + num_steps * self.warmup_step_size
        else:
            self.lr = self.base_lr * (self.warmup_steps ** 0.5) * (num_steps ** -0.5)

        return [self.lr for group in self.optimizer.param_groups]
