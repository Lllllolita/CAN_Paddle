from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from paddle.optimizer.lr import *
from paddlevision.optimizer.lr_scheduler import TwoStepCosineDecay

class Cosine(object):
    """
    Cosine learning rate decay
    lr = 0.05 * (math.cos(epoch * (math.pi / epochs)) + 1)
    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        epochs(int): total training epochs
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(self,
                 learning_rate,
                 step_each_epoch,
                 epochs1,
                 epochs2,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        super(Cosine, self).__init__()
        self.learning_rate = learning_rate
        self.T_max1 = step_each_epoch * epochs1
        self.T_max2 = step_each_epoch * epochs2
        self.last_epoch = last_epoch
        self.warmup_epoch = round(warmup_epoch * step_each_epoch)

    def __call__(self):
        learning_rate = TwoStepCosineDecay(
            learning_rate=self.learning_rate,
            T_max1=self.T_max1,
            T_max2=self.T_max2,
            last_epoch=self.last_epoch)
        if self.warmup_epoch > 0:
            learning_rate = LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_epoch,
                start_lr=0.0,
                end_lr=self.learning_rate,
                last_epoch=self.last_epoch)
        return learning_rate
