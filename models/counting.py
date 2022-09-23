import paddle
import paddle.nn as nn


class ChannelAtt(nn.Layer):
    def __init__(self, channel, reduction):
        super(ChannelAtt, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        w_attr_1, b_attr_1 = self._init_weights()
        w_attr_2, b_attr_2 = self._init_weights()
        self.fc = nn.Sequential(
                nn.Linear(channel, channel//reduction, weight_attr = w_attr_1, bias_attr=b_attr_1),
                nn.ReLU(),
                nn.Linear(channel//reduction, channel,  weight_attr = w_attr_2, bias_attr=b_attr_2),
                nn.Sigmoid())

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer = nn.initializer.KaimingUniform())
        bias_attr = paddle.ParamAttr(
            initializer = nn.initializer.KaimingUniform())
        return weight_attr, bias_attr

    def forward(self, x):
        b, c, _, _ = x.shape
        y = paddle.reshape(self.avg_pool(x),[b,c])
        y = paddle.reshape(self.fc(y),[b,c,1,1])
        return x * y

class CountingDecoder(nn.Layer):
    '''
    多尺度计数模块
    '''
    def __init__(self, in_channel, out_channel, kernel_size):
        super(CountingDecoder, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        w_attr_1, b_attr_1 = self._init_weights()

        self.trans_layer = nn.Sequential(
            nn.Conv2D(self.in_channel, 512, kernel_size=kernel_size, padding=kernel_size//2, weight_attr = w_attr_1,bias_attr=False),
            nn.BatchNorm2D(512))

        self.channel_att = ChannelAtt(512, 16)
        w_attr_2, b_attr_2 = self._init_weights()
        self.pred_layer = nn.Sequential(
            nn.Conv2D(512, self.out_channel, kernel_size=1, weight_attr = w_attr_2, bias_attr=False),
            nn.Sigmoid())

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer = nn.initializer.KaimingUniform())
        bias_attr = paddle.ParamAttr(
            initializer = nn.initializer.KaimingUniform())
        return weight_attr, bias_attr
    def forward(self, x, mask):
        b, c, h, w = x.shape
        x = self.trans_layer(x)
        x = self.channel_att(x)
        x = self.pred_layer(x)
        # mask可能用于数据增强
        if mask is not None:
            x = x * mask
    
        x = paddle.reshape(x,[b,self.out_channel, -1])
        x1 = paddle.sum(x, axis=-1)
  
        return x1,paddle.reshape(x,[b, self.out_channel, h, w])