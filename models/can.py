# encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle.nn as nn
from models.decoder import AttDecoder
from models.densenet import DenseNet
from models.counting import CountingDecoder as counting_decoder
from utils import *

class CAN(nn.Layer):
    def __init__(self, params=None):
        super(CAN, self).__init__()
        
        self.params = params
        self.use_label_mask = params['use_label_mask']

        # backbone
        self.encoder = DenseNet(params=self.params)

        # decoder
        self.in_channel = params['counting_decoder']['in_channel']
        self.out_channel = params['counting_decoder']['out_channel']

        # mscm
        self.counting_decoder1 = counting_decoder(self.in_channel, self.out_channel, 3)
        self.counting_decoder2 = counting_decoder(self.in_channel, self.out_channel, 5)

        self.decoder = AttDecoder(self.params)

        # 经过cnn后 长宽与原始尺寸比缩小的比例
        self.ratio = params['densenet']['ratio']

    def forward(self, images, images_mask, labels, labels_mask, is_train=True):
        cnn_features = self.encoder(images)

        counting_mask = images_mask[:, :, ::self.ratio, ::self.ratio]
        
        counting_preds1, _ = self.counting_decoder1(cnn_features, counting_mask)
        counting_preds2, _ = self.counting_decoder2(cnn_features, counting_mask)
        counting_preds = (counting_preds1 + counting_preds2) / 2

        word_probs, word_alphas = self.decoder(cnn_features, labels, counting_preds, images_mask, labels_mask, is_train=is_train)
        
        return word_probs, counting_preds,counting_preds1,counting_preds2