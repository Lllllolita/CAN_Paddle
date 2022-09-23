import paddle
import paddle.nn as nn
from utils.util_counting import gen_counting_label


class Loss(nn.Layer):
    '''
    CAN的loss由两部分组成
    word_average_loss：模型输出的符号准确率
    counting_loss：每类符号的计数损失
    '''
    def __init__(self, params=None):
        super(Loss, self).__init__()

        self.use_label_mask = params['use_label_mask']
        self.out_channel = params['counting_decoder']['out_channel']

        self.cross = nn.CrossEntropyLoss(reduction='none') if self.use_label_mask else nn.CrossEntropyLoss()
        self.counting_loss = nn.SmoothL1Loss(reduction='mean')

        self.ratio = params['densenet']['ratio']

    def forward(self, labels, labels_mask,word_probs, counting_preds, counting_preds1, counting_preds2):
  
        counting_labels = gen_counting_label(labels, self.out_channel, True)
        counting_loss = self.counting_loss(counting_preds1, counting_labels) + self.counting_loss(counting_preds2, counting_labels) \
                        + self.counting_loss(counting_preds, counting_labels)

        word_loss = self.cross(paddle.reshape(word_probs,[-1, word_probs.shape[-1]]), paddle.reshape(labels,[-1]))
        word_average_loss = paddle.sum(paddle.reshape(word_loss * labels_mask,[-1])) / (paddle.sum(labels_mask) + 1e-10) if self.use_label_mask else word_loss
        
        return word_average_loss, counting_loss