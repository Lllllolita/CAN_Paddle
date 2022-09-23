import numpy as np


class MaskGenerator:
    def __init__(self, height=481, width=2116):
        self.height = height
        self.width = width
    
    def forward(self, in_image):
        image = np.zeros((1, self.height, self.width))
        
        _, _, h, w = image.shape
        image[:, :h, :w] = in_image

        return image

class Words:
    '''
    输出序列相关类
    '''
    def __init__(self, word_path):
        with open(word_path) as f:
            words = f.readlines()
            print(f'共 {len(words)} 类符号。')
        self.words_dict = {words[i].strip(): i for i in range(len(words))}
        self.words_index_dict = {i: words[i].strip()
                                 for i in range(len(words))}

    def __len__(self):
        return len(self.words_dict)

    # 字符映射至索引
    def encode(self, labels):
        label_index = [self.words_dict[item] for item in labels]
        return label_index

    # 索引映射至字符
    def decode(self, label_index):
        label = ' '.join([self.words_index_dict[int(item)]
                         for item in label_index])
        return label