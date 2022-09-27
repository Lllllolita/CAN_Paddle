import os
import cv2
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math

from models.densenet import DenseNet
from models.attention import Attention
from models.decoder import PositionEmbeddingSine
from models.counting import CountingDecoder as counting_decoder

# 模型推理的时候用的
class Inference(nn.Layer):
    def __init__(self, params=None, draw_map=False):
        super(Inference, self).__init__()
        self.params = params
        self.draw_map = draw_map
        self.use_label_mask = params['use_label_mask']
        self.encoder = DenseNet(params=self.params)
        self.in_channel = params['counting_decoder']['in_channel']
        self.out_channel = params['counting_decoder']['out_channel']
        self.counting_decoder1 = counting_decoder(self.in_channel, self.out_channel, 3)
        self.counting_decoder2 = counting_decoder(self.in_channel, self.out_channel, 5)
        self.device = params['device']
        self.decoder = decoder_dict[params['decoder']['net']](params=self.params)

        """经过cnn后 长宽与原始尺寸比缩小的比例"""
        self.ratio = params['densenet']['ratio']

        with open(params['word_path']) as f:
            words = f.readlines()
            print(f'共 {len(words)} 类符号。')
        self.words_index_dict = {i: words[i].strip() for i in range(len(words))}
        self.cal_mae = nn.L1Loss(reduction='mean')
        self.cal_mse = nn.MSELoss(reduction='mean') 
    # @paddle.jit.to_static
    def forward(self, images, is_train=False):
    # def forward(self, images, labels, name, is_train=False):
        cnn_features = self.encoder(images)
    
        counting_preds1, _ = self.counting_decoder1(cnn_features, None)
        counting_preds2, _ = self.counting_decoder2(cnn_features, None)
        counting_preds = (counting_preds1 + counting_preds2) / 2

        word_probs = self.decoder(cnn_features, counting_preds, is_train=is_train)

        return word_probs


class AttDecoder(nn.Layer):
    def __init__(self, params):
        super(AttDecoder, self).__init__()
        self.params = params
        self.input_size = params['decoder']['input_size']
        self.hidden_size = params['decoder']['hidden_size']
        self.out_channel = params['encoder']['out_channel']
        self.attention_dim = params['attention']['attention_dim']
        self.dropout_prob = params['dropout']
        self.device = params['device']
        self.word_num = params['word_num']
        self.ratio = params['densenet']['ratio']

        self.init_weight = nn.Linear(self.out_channel, self.hidden_size)
        self.embedding = nn.Embedding(self.word_num, self.input_size)
        self.word_input_gru = nn.GRUCell(self.input_size, self.hidden_size)
      
        self.encoder_feature_conv = nn.Conv2D(self.out_channel, self.attention_dim, kernel_size=1)
        self.word_attention = Attention(params)

        self.word_state_weight = nn.Linear(self.hidden_size, self.hidden_size)
        self.word_embedding_weight = nn.Linear(self.input_size, self.hidden_size)
        self.word_context_weight = nn.Linear(self.out_channel, self.hidden_size)
        self.counting_context_weight = nn.Linear(self.word_num, self.hidden_size)
        self.word_convert = nn.Linear(self.hidden_size, self.word_num)
        
        if params['dropout']:
            self.dropout = nn.Dropout(params['dropout_ratio'])

    def forward(self, cnn_features, counting_preds, is_train=False):
       
        batch_size, _, height, width = cnn_features.shape
        
        image_mask = paddle.ones([batch_size, 1, height, width])

        cnn_features_trans = self.encoder_feature_conv(cnn_features)
        position_embedding = PositionEmbeddingSine(256, normalize=True)
        pos = position_embedding(cnn_features_trans, image_mask[:,0,:,:])
        cnn_features_trans = cnn_features_trans + pos


        word_alpha_sum = paddle.zeros([batch_size, 1, height, width])
        hidden = self.init_hidden(cnn_features, image_mask)
       
        word_embedding = self.embedding(paddle.ones([batch_size],dtype='int32'))
        counting_context_weighted = self.counting_context_weight(counting_preds)
        word_probs = []
        word=paddle.ones([1],dtype='int32')
        
        i = 0
        
        while i < 200:
            
            _,hidden = self.word_input_gru(word_embedding, hidden)
            
            word_context_vec, _, word_alpha_sum = self.word_attention(cnn_features, cnn_features_trans, hidden,
                                                                               word_alpha_sum, image_mask)
            
            current_state = self.word_state_weight(hidden)
            word_weighted_embedding = self.word_embedding_weight(word_embedding)
            word_context_weighted = self.word_context_weight(word_context_vec)
            
            if self.params['dropout']:
                word_out_state = self.dropout(current_state + word_weighted_embedding + word_context_weighted + counting_context_weighted) 
            else:
                word_out_state = current_state + word_weighted_embedding + word_context_weighted + counting_context_weighted

            word_prob = self.word_convert(word_out_state)

            #debug改的语句
            # import pdb; pdb.set_trace()
            _, word = paddle.topk(word_prob,k=1,axis=1)
            #只考虑一张图片输入
        
            word=paddle.to_tensor(word[0])
      
            word_embedding = self.embedding(word)
           
            #解码遇到eos就会停止
            #为了导出模型改的参数，因为他的变量可能固化为static，无法使用item
            if word.item() == 0:
                return word_probs
         
            word_probs.append(word)
            i+=1
            # print("i")
            # print(i)
            
        return word_probs

    def init_hidden(self, features, feature_mask):
        # average = (features * feature_mask).sum(-1).sum(-1) / feature_mask.sum(-1).sum(-1)
        average = paddle.sum(paddle.sum(features * feature_mask,axis=-1),axis=-1) /paddle.sum((paddle.sum(feature_mask,axis=-1)),axis=-1)
        average = self.init_weight(average)
        return paddle.tanh(average)


decoder_dict = {
    'AttDecoder': AttDecoder
}
