import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
'''densenet:
   ratio: 16
   growthRate: 24
   reduction: 0.5
   bottleneck: True
   use_dropout: True
'''

# DenseNet-B
class Bottleneck(nn.Layer):
    def __init__(self, nChannels, growthRate, use_dropout):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2D(interChannels)
        # 使用kaiming初始化
        # w_attr_1, b_attr_1 = self._init_weights()
        # self.conv1 = nn.Conv2D(nChannels, interChannels, kernel_size=1, weight_attr = w_attr_1, bias_attr=None)
        # 使用Xavier初始化
        self.conv1 = nn.Conv2D(nChannels, interChannels, kernel_size=1, bias_attr=None)
        self.bn2 = nn.BatchNorm2D(growthRate)

        # 使用kaiming初始化
        # w_attr_2, b_attr_2 = self._init_weights()
        # self.conv2 = nn.Conv2D(interChannels, growthRate, kernel_size=3, padding=1, weight_attr = w_attr_2, bias_attr=None)
        self.conv2 = nn.Conv2D(interChannels, growthRate, kernel_size=3, padding=1, bias_attr=None)
        # 使用Xavier初始化
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    # def _init_weights(self):
    #     weight_attr = paddle.ParamAttr(
    #         initializer = nn.initializer.KaimingUniform())
    #     bias_attr = paddle.ParamAttr(
    #         initializer = nn.initializer.KaimingUniform())
    #     return weight_attr, bias_attr

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.use_dropout:
            out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)))
        if self.use_dropout:
            out = self.dropout(out)
        out = paddle.concat([x, out], 1)
        return out


# single layer
class SingleLayer(nn.Layer):
    def __init__(self, nChannels, growthRate, use_dropout):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2D(nChannels)
        # 使用kaiming初始化
        # w_attr_1, b_attr_1 = self._init_weights()
        # self.conv1 = nn.Conv2D(nChannels, growthRate, kernel_size=3, padding=1, weight_attr = w_attr_1, bias_attr=False)
        # 使用Xavier初始化
        self.conv1 = nn.Conv2D(nChannels, growthRate, kernel_size=3, padding=1, bias_attr=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    # def _init_weights(self):
    #     weight_attr = paddle.ParamAttr(
    #         initializer = nn.initializer.KaimingUniform())
    #     bias_attr = paddle.ParamAttr(
    #         initializer = nn.initializer.KaimingUniform())
    #     return weight_attr, bias_attr

    def forward(self, x):
        out = self.conv1(F.relu(x))
        if self.use_dropout:
            out = self.dropout(out)
    
        out = paddle.concat([x, out], 1)
        return out


# transition layer
class Transition(nn.Layer):
    def __init__(self, nChannels, nOutChannels, use_dropout):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2D(nOutChannels)
        # 使用kaiming初始化
        # w_attr_1, b_attr_1 = self._init_weights()
        # self.conv1 = nn.Conv2D(nChannels, nOutChannels, kernel_size=1, weight_attr = w_attr_1, bias_attr=False)
        # 使用Xavier初始化
        self.conv1 = nn.Conv2D(nChannels, nOutChannels, kernel_size=1, bias_attr=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    # def _init_weights(self):
    #     weight_attr = paddle.ParamAttr(
    #         initializer = nn.initializer.KaimingUniform())
    #     bias_attr = paddle.ParamAttr(
    #         initializer = nn.initializer.KaimingUniform())
    #     return weight_attr, bias_attr

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.use_dropout:
            out = self.dropout(out)
        out = F.avg_pool2d(out, 2, ceil_mode=True,exclusive=False)
        return out


class DenseNet(nn.Layer):
    def __init__(self, params):
        super(DenseNet, self).__init__()
        # ratio: 16
        # growthRate: 24
        # reduction: 0.5
        growthRate = params['densenet']['growthRate']
        reduction = params['densenet']['reduction']
        bottleneck = params['densenet']['bottleneck']
        use_dropout = params['densenet']['use_dropout']

        nDenseBlocks = 16
        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2D(params['encoder']['input_channel'], nChannels, kernel_size=7, padding=3, stride=2, bias_attr=False)
       
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels, use_dropout)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels, use_dropout)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate, use_dropout))
            else:
                layers.append(SingleLayer(nChannels, growthRate, use_dropout))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
       
        out = F.max_pool2d(out, 2, ceil_mode=True)
        out = self.dense1(out)
        out = self.trans1(out)
        out = self.dense2(out)
        out = self.trans2(out)
        out = self.dense3(out)
        return out
