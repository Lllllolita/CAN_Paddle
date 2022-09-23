import math
from paddle.optimizer.lr import LRScheduler

class TwoStepCosineDecay(LRScheduler):

    def __init__(self,
                 learning_rate,
                 T_max1,
                 T_max2,
                 eta_min=0,
                 last_epoch=-1,
                 verbose=False):
        if not isinstance(T_max1, int):
            raise TypeError(
                "The type of 'T_max1' in 'CosineAnnealingDecay' must be 'int', but received %s."
                % type(T_max1))
        if not isinstance(T_max2, int):
            raise TypeError(
                "The type of 'T_max2' in 'CosineAnnealingDecay' must be 'int', but received %s."
                % type(T_max2))
        if not isinstance(eta_min, (float, int)):
            raise TypeError(
                "The type of 'eta_min' in 'CosineAnnealingDecay' must be 'float, int', but received %s."
                % type(eta_min))
        assert T_max1 > 0 and isinstance(
            T_max1, int), " 'T_max1' must be a positive integer."
        assert T_max2 > 0 and isinstance(
            T_max2, int), " 'T_max1' must be a positive integer."
        self.T_max1 = T_max1
        self.T_max2 = T_max2
        self.eta_min = float(eta_min)
        super(TwoStepCosineDecay, self).__init__(learning_rate, last_epoch,
                                                   verbose)

    def get_lr(self):

        if self.last_epoch <= self.T_max1:
            if self.last_epoch == 0:
                return self.base_lr
            elif (self.last_epoch - 1 - self.T_max1) % (2 * self.T_max1) == 0:
                return self.last_lr + (self.base_lr - self.eta_min) * (1 - math.cos(
                    math.pi / self.T_max1)) / 2

            return (1 + math.cos(math.pi * self.last_epoch / self.T_max1)) / (
                1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max1)) * (
                    self.last_lr - self.eta_min) + self.eta_min
        else:
            if (self.last_epoch - 1 - self.T_max2) % (2 * self.T_max2) == 0:
                return self.last_lr + (self.base_lr - self.eta_min) * (1 - math.cos(
                    math.pi / self.T_max2)) / 2

            return (1 + math.cos(math.pi * self.last_epoch / self.T_max2)) / (
                    1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max2)) * (
                           self.last_lr - self.eta_min) + self.eta_min

    def _get_closed_form_lr(self):
        if self.last_epoch <= self.T_max1:
            return self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(
            math.pi * self.last_epoch / self.T_max1)) / 2
        else:
            return self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(
            math.pi * self.last_epoch / self.T_max2)) / 2
