import cv2, paddle
import numpy as np

from difflib import SequenceMatcher


class Meter:
    def __init__(self, alpha=0.9):
        self.nums = []
        self.exp_mean = 0
        self.alpha = alpha

    @property
    def mean(self):
        return np.mean(self.nums)

    def add(self, num):
        if len(self.nums) == 0:
            self.exp_mean = num
        self.nums.append(num)
        self.exp_mean = self.alpha * self.exp_mean + (1 - self.alpha) * num


def cal_score(word_probs, word_label, mask):
    """
    :param word_probs: tensor
    :param word_label: tensor
    :param mask: tensor
    :return: float
    """
    line_right = 0
    if word_probs is not None:
        word_pred = word_probs.argmax(2)
    # print(word_probs)
    word_scores = [SequenceMatcher(None, s1[:int(np.sum(s3))], s2[:int(np.sum(s3))], autojunk=False).ratio() * (
                len(s1[:int(np.sum(s3))]) + len(s2[:int(np.sum(s3))])) / len(s1[:int(np.sum(s3))]) / 2
                   for s1, s2, s3 in zip(word_label.cpu().detach().numpy(), word_pred.cpu().detach().numpy(), mask.cpu().detach().numpy())]
    batch_size = len(word_scores)
    for i in range(batch_size):
        if word_scores[i] == 1:
            line_right += 1
    ExpRate = line_right / batch_size #float
    word_scores = np.mean(word_scores) #float
    return word_scores, ExpRate


def draw_attention_map(image, attention):
    h, w = image.shape
    attention = cv2.resize(attention, (w, h))
    attention_heatmap = ((attention - np.min(attention)) / (np.max(attention) - np.min(attention))*255).astype(np.uint8)
    attention_heatmap = cv2.applyColorMap(attention_heatmap, cv2.COLORMAP_JET)
    image_new = np.stack((image, image, image), axis=-1).astype(np.uint8)
    attention_map = cv2.addWeighted(attention_heatmap, 0.4, image_new, 0.6, 0.)
    return attention_map


def draw_counting_map(image, counting_attention):
    h, w = image.shape
    counting_attention = paddle.clip(counting_attention, 0.0, 1.0).numpy()
    counting_attention = cv2.resize(counting_attention, (w, h))
    counting_attention_heatmap = (counting_attention * 255).astype(np.uint8)
    counting_attention_heatmap = cv2.applyColorMap(counting_attention_heatmap, cv2.COLORMAP_JET)
    image_new = np.stack((image, image, image), axis=-1).astype(np.uint8)
    counting_map = cv2.addWeighted(counting_attention_heatmap, 0.4, image_new, 0.6, 0.)
    return counting_map


def cal_distance(word1, word2):
    m = len(word1)
    n = len(word2)
    if m*n == 0:
        return m+n
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            a = dp[i-1][j] + 1
            b = dp[i][j-1] + 1
            c = dp[i-1][j-1]
            if word1[i-1] != word2[j-1]:
                c += 1
            dp[i][j] = min(a, b, c)
    return dp[m][n]


def compute_edit_distance(prediction, label):
    prediction = prediction.strip().split(' ')
    label = label.strip().split(' ')
    distance = cal_distance(prediction, label)
    return distance
    
