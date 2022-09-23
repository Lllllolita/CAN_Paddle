import paddle
import paddle.nn as nn


class Attention(nn.Layer):
    def __init__(self, params):
        super(Attention, self).__init__()
        self.params = params
        # hidden_size :256
        self.hidden = params['decoder']['hidden_size']
        # attention_dim :512
        self.attention_dim = params['attention']['attention_dim']

        w_attr_1, b_attr_1 = self._init_weights()
        self.hidden_weight = nn.Linear(self.hidden, self.attention_dim, weight_attr = w_attr_1, bias_attr=b_attr_1)
        # spatial attention
        w_attr_2, b_attr_2 = self._init_weights()
        self.attention_conv = nn.Conv2D(1, 512, kernel_size=11, padding=5, weight_attr = w_attr_2, bias_attr=False)

        w_attr_3, b_attr_3 = self._init_weights()
        self.attention_weight = nn.Linear(512, self.attention_dim, weight_attr = w_attr_3, bias_attr=False)

        w_attr_4, b_attr_4 = self._init_weights()
        self.alpha_convert = nn.Linear(self.attention_dim, 1, weight_attr = w_attr_4, bias_attr=b_attr_4)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer = nn.initializer.KaimingUniform())
        bias_attr = paddle.ParamAttr(
            initializer = nn.initializer.KaimingUniform())
        return weight_attr, bias_attr

    def forward(self, cnn_features, cnn_features_trans, hidden, alpha_sum, image_mask=None):
        # W_h*h_t:b 256 -> b 512(b attention_dim)
        query = self.hidden_weight(hidden)
        # A(coverage_alpha):b 1 H W -> b 512 H W  用于通道变换
        alpha_sum_trans = self.attention_conv(alpha_sum)
        # W_a*A: b 512 h w -> b h w 512-> b h w attention_dim
       
        coverage_alpha = self.attention_weight(paddle.transpose(alpha_sum_trans,[0,2,3,1]))
        # w_T*tanh( W_a*A + W_h*h_t + T + P) + b
        # tensor的加法遵循广播机制 所以相加之后的大小为 b h w 512
        
        alpha_score = paddle.tanh(paddle.unsqueeze(query,[1, 2]) + coverage_alpha + paddle.transpose(cnn_features_trans,[0,2,3,1]))
        
        # b h w 512 -> b h w 1
        energy = self.alpha_convert(alpha_score)
        # 进行归一化
        # energy = energy - energy.max()
        # energy = energy - paddle.max(energy,keepdim=True)
        # energy = energy - paddle.max(energy,keepdim=False)
        energy = energy - energy.max()
        # b h w 1 -> b h w 
        # energy_exp = paddle.exp(energy.squeeze(-1))
        energy_exp = paddle.exp(paddle.squeeze(energy,-1))
        # image_mask :b 1 max_h max_w
        # * 应该是对位相乘 b h w ->b max_h max_w
        if image_mask is not None:
            # energy_exp = energy_exp * image_mask.squeeze(1)
            energy_exp = energy_exp * paddle.squeeze(image_mask,1)

       
        alpha = energy_exp / (paddle.unsqueeze(paddle.sum(paddle.sum(energy_exp,-1),-1),[1,2]) + 1e-10)
      
        alpha_sum = paddle.unsqueeze(alpha,1) + alpha_sum
       
        context_vector = paddle.sum(paddle.sum((paddle.unsqueeze(alpha,1) * cnn_features),-1),-1)
        return context_vector, alpha, alpha_sum