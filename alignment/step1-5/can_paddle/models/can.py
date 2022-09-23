import paddle
import paddle.nn as nn
from can_paddle.models.decoder import AttDecoder
from can_paddle.models.densenet import DenseNet
from can_paddle.models.counting import CountingDecoder as counting_decoder


class CAN(nn.Layer):
    def __init__(self, params=None):
        super(CAN, self).__init__()
        #初始化can需要载入参数
        self.params = params
        self.use_label_mask = params['use_label_mask']
        #backbone
        self.encoder = DenseNet(params=self.params)
        #decoder
        self.in_channel = params['counting_decoder']['in_channel']
        self.out_channel = params['counting_decoder']['out_channel']
        #mscm
        self.counting_decoder1 = counting_decoder(self.in_channel, self.out_channel, 3)
        self.counting_decoder2 = counting_decoder(self.in_channel, self.out_channel, 5)

        self.decoder = AttDecoder(self.params)

        """经过cnn后 长宽与原始尺寸比缩小的比例"""
        self.ratio = params['densenet']['ratio']

    def forward(self, images, images_mask, labels, labels_mask, is_train=True):
        cnn_features = self.encoder(images)
        counting_mask = images_mask[:, :, ::self.ratio, ::self.ratio]
        #每个bacth每个labels的数量，输入ground truth

        counting_preds1, _ = self.counting_decoder1(cnn_features, counting_mask)
        counting_preds2, _ = self.counting_decoder2(cnn_features, counting_mask)
        counting_preds = (counting_preds1 + counting_preds2) / 2

        word_probs, word_alphas = self.decoder(cnn_features, labels, counting_preds, images_mask, labels_mask, is_train=is_train)
        
        return word_probs, counting_preds,counting_preds1,counting_preds2