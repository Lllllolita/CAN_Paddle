import time
import numpy as np
import pickle as pkl

import paddle
from paddle.io import Dataset, DistributedBatchSampler, DataLoader


class HMERDataset(Dataset):
    '''
    构造数据集类
    '''
    def __init__(self, params, image_path, label_path, words, is_train=True, use_aug=False):
        super(HMERDataset, self).__init__()
        if image_path.endswith('.pkl'):
            with open(image_path, 'rb') as f:
                self.images = pkl.load(f)
        elif image_path.endswith('.list'):
            with open(image_path, 'r') as f:
                lines = f.readlines()
            self.images = {}
            print(f'data files: {lines}')
            for line in lines:
                name = line.strip()
                print(f'loading data file: {name}')
                start = time.time()
                with open(name, 'rb')as f:
                    images = pkl.load(f)
                self.images.update(images)
                print(f'loading {name} cost: {time.time()-start:.2f} seconds!')
        with open(label_path, 'r') as f:
            self.labels = f.readlines()

        self.words = words
        self.is_train = is_train
        self.params = params

    def __len__(self):
        assert len(self.images) == len(self.labels)
        return len(self.labels)

    def __getitem__(self, idx):
        name, *labels = self.labels[idx].strip().split()
        name = name.split('.')[0] if name.endswith('.jpg') else name
        image = self.images[name]
        image = np.expand_dims((255-image)/255,axis=0)
        labels.append('eos')
        words = np.array(self.words.encode(labels))
        return image, words


def get_crohme_dataset(params):
    '''
    获取数据集方法
    '''
    words = Words(params['word_path'])
    params['word_num'] = len(words)
    print(f"训练数据路径 images: {params['train_image_path']} labels: {params['train_label_path']}")
    print(f"验证数据路径 images: {params['eval_image_path']} labels: {params['eval_label_path']}")

    train_dataset = HMERDataset(params, params['train_image_path'], params['train_label_path'], words, is_train=True)
    eval_dataset = HMERDataset(params, params['eval_image_path'], params['eval_label_path'], words, is_train=False)

    # 构造采样器，支持单机多卡情形
    train_sampler = DistributedBatchSampler(dataset=train_dataset, batch_size=params["batch_size"], shuffle=True,drop_last=False)
    eval_sampler = DistributedBatchSampler(dataset=eval_dataset, batch_size=1, shuffle=True, drop_last=False)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']])
    eval_loader = DataLoader(eval_dataset, batch_sampler=eval_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']])

    print(f'train dataset: {len(train_dataset)} train steps: {len(train_loader)} '
          f'eval dataset: {len(eval_dataset)} eval steps: {len(eval_loader)} ')
    return train_loader, eval_loader


def collate_fn(batch_images):
    '''
    数据集加载前预处理（batch内统一尺寸）
    '''
    max_width, max_height, max_length = 0, 0, 0
    batch, channel = len(batch_images), batch_images[0][0].shape[0]
    proper_items = []
    for item in batch_images:
        if item[0].shape[1] * max_width > 1600 * 320 or item[0].shape[2] * max_height > 1600 * 320:
            continue
        max_height = item[0].shape[1] if item[0].shape[1] > max_height else max_height
        max_width = item[0].shape[2] if item[0].shape[2] > max_width else max_width
        max_length = item[1].shape[0] if item[1].shape[0] > max_length else max_length
        proper_items.append(item)

    images, image_masks = np.zeros((len(proper_items), channel, max_height, max_width)), np.zeros(
        (len(proper_items), 1, max_height, max_width))
    labels, labels_masks = np.zeros((len(proper_items), max_length), dtype='int64'), np.zeros(
        (len(proper_items), max_length))

    for i in range(len(proper_items)):
        _, h, w = proper_items[i][0].shape
        images[i][:, :h, :w] = proper_items[i][0]
        image_masks[i][:, :h, :w] = 1
        l = proper_items[i][1].shape[0]
        labels[i][:l] = proper_items[i][1]
        labels_masks[i][:l] = 1
    return images, image_masks, labels, labels_masks


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

collate_fn_dict = {
    'collate_fn': collate_fn
}

