from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from paddlevision.optimizer.learning_rate import *
import copy
import paddle

__all__ = ['build_lr_scheduler']

def build_lr_scheduler(lr, step_each_epoch, epochs1, epochs2, warmup_epoch):
    """
        Args:
            lr(float): initial learning rate
            step_each_epoch(int): steps each epoch
            epochs(int): total training epochs
            warmup_epoch: start warmup
            last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        """
    lr = Cosine(lr,
                step_each_epoch,
                epochs1,
                epochs2,
                warmup_epoch = warmup_epoch,
                last_epoch=-1)()
    return lr

