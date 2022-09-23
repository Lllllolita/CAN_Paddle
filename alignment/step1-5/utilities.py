import numpy as np
import paddle
import torch
import os, yaml, random

import paddlevision.datasets.dataset as paddlevision
from can_ref import dataset as torchvision
from difflib import SequenceMatcher
import math

random.seed(666)
np.random.seed(666)
paddle.seed(666)
torch.manual_seed(666)
torch.cuda.manual_seed(666)

def gen_fake_data():
    '''
    构造假数据
    数据：原始数据集包含的图像大小不一，在输入前根据batch内最大尺寸进行扩张，这里选择的大小为数据集平均大小
    标签：共111类符号，假设符号长度均为24（实际中对长度更短的序列进行补0操作）
    mask: 由于数据和标签都是不确定大小的数据，因此在全部设置为相同大小后通过mask标识有效数据区域
    '''
    fake_data = np.random.rand(8, 1, 103, 314).astype(np.float32) - 0.5
    fake_data_mask = np.ones((8, 1, 103, 314))
    fake_label = np.random.randint(0, high=110, size=(8,24), dtype="int64")
    fake_label_mask = np.ones((8, 24))
    np.save("fake_data.npy", fake_data)
    np.save("fake_label.npy", fake_label)
    np.save("fake_data_mask.npy", fake_data_mask)
    np.save("fake_label_mask.npy", fake_label_mask)


def build_paddle_data_pipeline(image_path, label_path, words):
    '''
    构建paddle数据集
    '''

    dataset_test = paddlevision.HMERDataset({}, image_path, label_path, words)

    # 固定batch size
    data_loader_test = paddle.io.DataLoader(
        dataset_test,
        num_workers=0,
        batch_size=8,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn_dict['collate_fn_paddle'])

    return dataset_test, data_loader_test


def build_torch_data_pipeline(image_path, label_path, words):
    '''
    构建torch数据集
    '''

    dataset_test = torchvision.HMERDataset({}, image_path, label_path, words)

    # 固定batch size，关闭shuffle
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=8,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=collate_fn_dict['collate_fn_torch'])

    return dataset_test, data_loader_test


def evaluate(image, image_mask, label, label_mask, model, tag, reprod_logger, dir):
    '''
    前向对齐
    :@param image: 图像
    :@param image_mask: 图像遮罩
    :@param labels: 标签
    :@param label_mask: 标签遮罩
    :@param model: 模型
    :@param tag
    :@param reprod_logger
    '''
    model.eval()
    word_probs, _, _, _ = model(image, image_mask, label, label_mask)
    if tag=='ref':
        WordRate, ExpRate = cal_score_torch(word_probs, label, label_mask)
    else:
        WordRate, ExpRate = cal_score_paddle(word_probs, label, label_mask)

    reprod_logger.add("WordRate", np.array(WordRate))
    reprod_logger.add("ExpRate", np.array(ExpRate))
    reprod_logger.save(os.path.join(dir,"metric_{}.npy".format(tag)))

def cal_score_paddle(word_probs, word_label, mask):
    """
    :param word_probs: tensor
    :param word_label: tensor
    :param mask: tensor
    :return: float
    """
    line_right = 0
    if word_probs is not None:
        word_pred = word_probs.argmax(2)
    word_scores = [SequenceMatcher(None, s1[:int(np.sum(s3))], s2[:int(np.sum(s3))], autojunk=False).ratio() * (
            len(s1[:int(np.sum(s3))]) + len(s2[:int(np.sum(s3))])) / len(s1[:int(np.sum(s3))]) / 2
                   for s1, s2, s3 in zip(word_label.cpu().detach().numpy(), word_pred.cpu().detach().numpy(), mask.cpu().detach().numpy())]
    batch_size = len(word_scores)
    for i in range(batch_size):
        if word_scores[i] == 1:
            line_right += 1
    ExpRate = line_right / batch_size  # float
    word_scores = np.mean(word_scores)  # float
    return word_scores, ExpRate


def cal_score_torch(word_probs, word_label, mask):
    """
    :param word_probs: tensor
    :param word_label: tensor
    :param mask: tensor
    :return: float
    """
    line_right = 0
    if word_probs is not None:
        _, word_pred = word_probs.max(2)
    word_scores = [SequenceMatcher(None, s1[:int(np.sum(s3))], s2[:int(np.sum(s3))], autojunk=False).ratio() * (
            len(s1[:int(np.sum(s3))]) + len(s2[:int(np.sum(s3))])) / len(s1[:int(np.sum(s3))]) / 2
                   for s1, s2, s3 in
                   zip(word_label.cpu().detach().numpy(), word_pred.cpu().detach().numpy(), mask.cpu().detach().numpy())]
    batch_size = len(word_scores)
    for i in range(batch_size):
        if word_scores[i] == 1:
            line_right += 1

    ExpRate = line_right / batch_size  # float
    word_scores = np.mean(word_scores)  # float
    return word_scores, ExpRate


def train_paddle(model, criterion, optimizer, lr_scheduler, max_iter, inputs, inputs_mask, labels, labels_mask, reprod_logger):
    '''
    学习率和loss对齐——paddle训练一个轮次
    '''

    for idx in range(max_iter):
        image = paddle.to_tensor(inputs,dtype='float32')
        target = paddle.to_tensor(labels, dtype='int64')
        image_mask = paddle.to_tensor(inputs_mask, dtype='float32')
        target_mask = paddle.to_tensor(labels_mask, dtype='int64')

        lr_scheduler.step()
        optimizer.clear_grad()

        word_probs, counting_preds, counting_preds1, counting_preds2 = model(image,
                                                                             image_mask,
                                                                             target,
                                                                             target_mask)

        word_loss, counting_loss = criterion(target, target_mask, word_probs, counting_preds, counting_preds1, counting_preds2)
        loss = word_loss + counting_loss

        reprod_logger.add("loss_{}".format(idx), loss.detach().numpy())
        reprod_logger.add("lr_{}".format(idx), np.array(lr_scheduler.get_lr()))

        loss.backward()
        optimizer.step()

    reprod_logger.save("alignment/result/losses_paddle.npy")


def update_lr(optimizer, current_epoch, current_step, steps, epochs, initial_lr):
    if current_epoch < 1:
        new_lr = initial_lr / steps * (current_step + 1)
    elif 1 <= current_epoch <= 200:
        new_lr = 0.5 * (1 + math.cos((current_step + 1 + (current_epoch - 1) * steps) * math.pi / (200 * steps))) * initial_lr
    else:
        new_lr = 0.5 * (1 + math.cos((current_step + 1 + (current_epoch - 1) * steps) * math.pi / (epochs * steps))) * initial_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def train_torch(model, criterion, optimizer, lr, max_iter, inputs, inputs_mask, labels, labels_mask, gradient, reprod_logger,torch_device):
    '''
    学习率和loss对齐——torch训练一个轮次
    '''
    for idx in range(max_iter):
        image = torch.tensor(inputs, dtype=torch.float32).cuda()
        target = torch.tensor(labels, dtype=torch.int64).cuda()
        image_mask = torch.tensor(inputs_mask, dtype=torch.float32).cuda()
        target_mask = torch.tensor(labels_mask, dtype=torch.int64).cuda()
        model = model.cuda()

        update_lr(optimizer, 0, idx, max_iter, 240, lr)
        optimizer.zero_grad()
        word_probs, counting_preds, counting_preds1, counting_preds2 = model(image,
                                                                             image_mask,
                                                                             target,
                                                                             target_mask)

        word_loss, counting_loss = criterion(target, target_mask,word_probs, counting_preds, counting_preds1, counting_preds2)
        loss = word_loss + counting_loss

        reprod_logger.add("loss_{}".format(idx), loss.cpu().detach().numpy())
        reprod_logger.add("lr_{}".format(idx), np.array(optimizer.param_groups[0]['lr']))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient)
        optimizer.step()
    reprod_logger.save("alignment/result/losses_ref.npy")

# 加载配置文件
def load_config(yaml_path):
    try:
        with open(yaml_path, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    except:
        print('尝试UTF-8编码....')
        print(yaml_path)
        with open(yaml_path, 'r', encoding='UTF-8') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    return params

def collate_fn_paddle(batch_images):
    '''
    数据集构建预处理（统一图像尺寸）
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

    images, image_masks = np.zeros((len(proper_items), channel, max_height, max_width),dtype='float32'), np.zeros((len(proper_items), 1, max_height, max_width), dtype='float32')
    labels, labels_masks = np.zeros((len(proper_items), max_length), dtype='int64'), np.zeros((len(proper_items), max_length))

    for i in range(len(proper_items)):
        _, h, w = proper_items[i][0].shape
        images[i][:, :h, :w] = proper_items[i][0]
        image_masks[i][:, :h, :w] = 1
        l = proper_items[i][1].shape[0]
        labels[i][:l] = proper_items[i][1]
        labels_masks[i][:l] = 1
    return images, image_masks, labels, labels_masks

def collate_fn_torch(batch_images):
    '''
    数据集构建预处理（统一图像尺寸）
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

    images, image_masks = torch.zeros((len(proper_items), channel, max_height, max_width)), torch.zeros((len(proper_items), 1, max_height, max_width))
    labels, labels_masks = torch.zeros((len(proper_items), max_length)).long(), torch.zeros((len(proper_items), max_length))

    for i in range(len(proper_items)):
        _, h, w = proper_items[i][0].shape
        images[i][:, :h, :w] = proper_items[i][0]
        image_masks[i][:, :h, :w] = 1
        l = proper_items[i][1].shape[0]
        labels[i][:l] = proper_items[i][1]
        labels_masks[i][:l] = 1
    return images, image_masks, labels, labels_masks

# paddle不支持子进程gpu Tensor运算，故此处做区分

collate_fn_dict = {
    'collate_fn_paddle': collate_fn_paddle,
    'collate_fn_torch': collate_fn_torch
}