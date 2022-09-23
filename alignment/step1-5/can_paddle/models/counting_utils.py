import paddle
import numpy as np

#难道是加载counitig的gt使用的文件
def gen_counting_label(labels, channel, tag):
    b , t =labels.shape
    counting_labels = np.zeros([b, channel]) 
    
    if tag:
        ignore = [0, 1, 107, 108, 109, 110]
    else:
        ignore = []
    for i in range(b):
        for j in range(t):
            k = labels[i][j]
            if k in ignore:
                continue
            else:
                counting_labels[i][k] += 1
    counting_labels = paddle.to_tensor(counting_labels,dtype='float32')
    return counting_labels
  
