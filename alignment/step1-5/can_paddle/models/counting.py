import paddle
import paddle.nn as nn


class ChannelAtt(nn.Layer):
    def __init__(self, channel, reduction):
        super(ChannelAtt, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel//reduction),
                nn.ReLU(),
                nn.Linear(channel//reduction, channel),
                nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).reshape([b,c])
        y = self.fc(y).reshape([b,c,1,1])

        return x * y

class CountingDecoder(nn.Layer):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(CountingDecoder, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.trans_layer = nn.Sequential(
            nn.Conv2D(self.in_channel, 512, kernel_size=kernel_size, padding=kernel_size//2, bias_attr=False),
            nn.BatchNorm2D(512))
        self.channel_att = ChannelAtt(512, 16)
        self.pred_layer = nn.Sequential(
            nn.Conv2D(512, self.out_channel, kernel_size=1, bias_attr=False),
            nn.Sigmoid())

    def forward(self, x, mask):
        b, c, h, w = x.shape

        x = self.trans_layer(x)
        x = self.channel_att(x)

        x = self.pred_layer(x)
        # mask可能用于数据增强
        if mask is not None:
            x = x * mask
        x = x.reshape((b,self.out_channel, -1))
        x1 = paddle.sum(x, axis=-1)
  
        return x1, x.reshape((b, self.out_channel, h, w))