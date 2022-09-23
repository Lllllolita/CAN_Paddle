import paddle
import paddle.nn as nn
from can_paddle.models.attention import Attention
import math


class PositionEmbeddingSine(nn.Layer):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        y_embed = paddle.cumsum(mask,1, dtype='float32')
        x_embed = paddle.cumsum(mask,2, dtype='float32')

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = paddle.arange(self.num_pos_feats, dtype='float32')
        # dim_d = paddle.expand(paddle.to_tensor(2), dim_t.shape)
        dim_t = self.temperature ** (2 * ((dim_t / 2).floor()) / self.num_pos_feats)

        pos_x = paddle.unsqueeze(x_embed,[3]) / dim_t
        pos_y = paddle.unsqueeze(y_embed,[3]) / dim_t
       
        pos_x = paddle.flatten(paddle.stack([paddle.sin(pos_x[:, :, :, 0::2]), paddle.cos(pos_x[:, :, :, 1::2])], axis=4),3)
        pos_y = paddle.flatten(paddle.stack([paddle.sin(pos_y[:, :, :, 0::2]), paddle.cos(pos_y[:, :, :, 1::2])], axis=4),3)

        pos = paddle.transpose(paddle.concat([pos_y, pos_x], axis=3),[0, 3, 1, 2])
       
        return pos


class AttDecoder(nn.Layer):
    def __init__(self, params):
        super(AttDecoder, self).__init__()
        self.params = params
        #256
        self.input_size = params['decoder']['input_size']
        #256
        self.hidden_size = params['decoder']['hidden_size']
        #684
        self.out_channel = params['encoder']['out_channel']
        #512
        self.attention_dim = params['attention']['attention_dim']
        self.dropout_prob = params['dropout']
        self.device = params['device']
        #
        self.word_num = params['word_num']
        #111(symbol class)
        self.counting_num = params['counting_decoder']['out_channel']

        paddle.device.set_device(self.device)

        """经过cnn后 长宽与原始尺寸比缩小的比例"""
        self.ratio = params['densenet']['ratio']

        # init hidden state
        self.init_weight = nn.Linear(self.out_channel, self.hidden_size)
        # word embedding
        # word_num , class -> 
        self.embedding = nn.Embedding(self.word_num, self.input_size)
        # word gru
        self.word_input_gru = nn.GRUCell(self.input_size, self.hidden_size)
        # attention
        self.word_attention = Attention(params)
        self.encoder_feature_conv = nn.Conv2D(self.out_channel, self.attention_dim,
                                              kernel_size=params['attention']['word_conv_kernel'],
                                              padding=params['attention']['word_conv_kernel']//2)

        self.word_state_weight = nn.Linear(self.hidden_size, self.hidden_size)
        self.word_embedding_weight = nn.Linear(self.input_size, self.hidden_size)
        self.word_context_weight = nn.Linear(self.out_channel, self.hidden_size)
        self.counting_context_weight = nn.Linear(self.counting_num, self.hidden_size)
        self.word_convert = nn.Linear(self.hidden_size, self.word_num)

        if params['dropout']:
            self.dropout = nn.Dropout(params['dropout_ratio'])
        
    def forward(self, cnn_features, labels, counting_preds, images_mask, labels_mask, is_train=True):
        batch_size, num_steps = labels.shape # b, t
        height, width = cnn_features.shape[2:]
        # b,t, word_num(最大词长，应为200)
        word_probs = paddle.zeros((batch_size, num_steps, self.word_num))
        images_mask = images_mask[:, :, ::self.ratio, ::self.ratio]
        # coverage attention
        # b,1,h,w
        word_alpha_sum = paddle.zeros((batch_size, 1, height, width))
        # b,t,h,w
        word_alphas = paddle.zeros((batch_size, num_steps, height, width))
        # b 256
        hidden = self.init_hidden(cnn_features, images_mask)
        # (WcC)counting vector input :b c -> b 256
        counting_context_weighted = self.counting_context_weight(counting_preds)
        # 1x1 conv
        cnn_features_trans = self.encoder_feature_conv(cnn_features)
        # H W 512
        position_embedding = PositionEmbeddingSine(256, normalize=True)
        # image mask dimense:b c h w?
        # b c h w -> 
        pos= position_embedding(cnn_features_trans, images_mask[:,0,:,:])
        # 生成位置编码之后 在原来的feature map上加入位置信息
        cnn_features_trans = cnn_features_trans + pos

        z_pred = paddle.ones(counting_preds.shape)
        z = self.counting_context_weight(z_pred)

        if is_train:
            #
            for i in range(num_steps):
                # b i-1  将前一个输出结果进行嵌入，embedding(y_t-1)
                # 输出的维度[symbol class hidden dimense]
                word_embedding = self.embedding(labels[:, i-1]) if i else self.embedding(paddle.ones([batch_size], dtype='int64'))
               
                _, hidden = self.word_input_gru(word_embedding, hidden)
                if i == 1 :
                    y = hidden
                # 输出 context_vector ,word_attention ,word_attention_sum
                word_context_vec, word_alpha, word_alpha_sum = self.word_attention(cnn_features, cnn_features_trans, hidden,word_alpha_sum, images_mask)
                # p(y_t)=softmax(wTo(WcC + WvV + Wtht + WeE) + bo )
                # Wtht                                          
                current_state = self.word_state_weight(hidden)
                # WeE
                word_weighted_embedding = self.word_embedding_weight(word_embedding)
                # WvV
                word_context_weighted = self.word_context_weight(word_context_vec)

                if self.params['dropout']:
                    # 对一些参数进行drop out (Wtht+WeE+WvV+WcC) 防止过拟合
                    word_out_state = self.dropout(current_state + word_weighted_embedding + word_context_weighted + counting_context_weighted)
                else:
                    word_out_state = current_state + word_weighted_embedding + word_context_weighted + counting_context_weighted

                word_prob = self.word_convert(word_out_state)
                word_probs[:, i] = word_prob
                word_alphas[:, i] = word_alpha
                #返回的是未经过softmax的word_probs和word_alpha
        else:
            #如果是前向的推理,模型进行推理的时候使用，并不进行梯度的计算
          
            word_embedding = self.embedding(paddle.ones([batch_size], dtype='int64'))
            for i in range(num_steps):
                #ht
                hidden = self.word_input_gru(word_embedding, hidden)
                # Vc,attention ,attention_sum
                word_context_vec, word_alpha, word_alpha_sum = self.word_attention(cnn_features, cnn_features_trans, hidden,word_alpha_sum, images_mask)
                #Wtht
                current_state = self.word_state_weight(hidden)
                #WeE
                word_weighted_embedding = self.word_embedding_weight(word_embedding)
                #WvV
                word_context_weighted = self.word_context_weight(word_context_vec)

                if self.params['dropout']:
                    word_out_state = self.dropout(current_state + word_weighted_embedding + word_context_weighted + counting_context_weighted)
                else:
                    word_out_state = current_state + word_weighted_embedding + word_context_weighted + counting_context_weighted
                # b word_num 
                word_prob = self.word_convert(word_out_state)
                # 输出word_prob在第二维的最大值,返回 max_value,max_index
                # b word_num
                # k 表示最大值的数量
                # _, word = paddle.topk(word_prob,k=1,axis=1)
                word = word_prob.argmax(1)
                word_embedding = self.embedding(word)
                word_probs[:, i] = word_prob
                word_alphas[:, i] = word_alpha
        return word_probs, word_alphas

    def init_hidden(self, features, feature_mask):
        # merge mask b c h w ->b c 
        average = (features * feature_mask).sum(axis=-1).sum(axis=-1) / feature_mask.sum(axis=-1).sum(axis=-1)
        # to hide dimense
        # b c  -> b 256
        average = self.init_weight(average)
        return paddle.tanh(average)
