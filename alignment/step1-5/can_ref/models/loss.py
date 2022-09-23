import torch
import torch.nn as nn
from can_ref.counting_utils import gen_counting_label

class CAN_Loss(nn.Module):
    def __init__(self, params=None):
        super(CAN_Loss, self).__init__()
        self.use_label_mask = params['use_label_mask']
        self.out_channel = params['counting_decoder']['out_channel']
        self.cross = nn.CrossEntropyLoss(reduction='none') if self.use_label_mask else nn.CrossEntropyLoss()
        self.counting_loss = nn.SmoothL1Loss(reduction='mean')
   

        """经过cnn后 长宽与原始尺寸比缩小的比例"""
        self.ratio = params['densenet']['ratio']

    def forward(self, labels, labels_mask, word_probs, counting_preds, counting_preds1, counting_preds2):
        #每个bacth每个labels的数量，应该喂入的是labels的gt
        counting_labels = gen_counting_label(labels, self.out_channel, True)

        counting_loss = self.counting_loss(counting_preds1, counting_labels) + self.counting_loss(counting_preds2, counting_labels) \
                        + self.counting_loss(counting_preds, counting_labels)

        word_loss = self.cross(word_probs.contiguous().view(-1, word_probs.shape[-1]), labels.view(-1))
        
        word_average_loss = (word_loss * labels_mask.view(-1)).sum() / (labels_mask.sum() + 1e-10) if self.use_label_mask else word_loss
        
        return word_average_loss, counting_loss