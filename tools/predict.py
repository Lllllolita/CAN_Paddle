import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.static import InputSpec
import pickle as pkl
import random
import yaml
import os
import sys
import numpy as np
import cv2
from models.infer_model import Inference as infer_paddle
from utils.util import load_config
from reprod_log import ReprodLogger
from paddlevision.datasets.dataset import Words

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))


def get_args(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(
        description='PaddlePaddle Can(OCR) Predict', add_help=add_help)
    parser.add_argument('--pretrained', default="./test_model/predict.pdparams", help='pretrained model')
    parser.add_argument('--device', default='cpu', help='device')
    parser.add_argument('--img_path', default='./test_images/test_example/test_01.jpeg', help='path where to save')
    parser.add_argument('--is_model_key', default=False, help='is_model_key')
    parser.add_argument('--config_file', default="./config.yaml", help='config_file')
    args = parser.parse_args()
    return args


@paddle.no_grad()
def main(args):
    # define model
    params=load_config(args.config_file)
   
    if args.device == 'gpu':
       # print(paddle.device.get_available_device())
       # assert len(paddle.device.get_available_device()) >= 1, "there are not available gpu device !."
       assert paddle.device.get_device()!='cpu',"there are not available gpu device"
       #devices = paddle.device.get_available_device().remove('cpu')
       # device = devices[random.randint(0,len(devices)-1)]
       device=paddle.device.get_device()
    else :
        device = 'cpu'
    params['device'] = device
    paddle.device.set_device(device)
    model=infer_paddle(params)
    if args.is_model_key:
        layer_state_dict=paddle.load(args.pretrained)['model']
    else :
        layer_state_dict=paddle.load(args.pretrained)

    model.set_state_dict(layer_state_dict)
    model.eval()
    

    # define transforms
    if args.img_path.endswith('.jpg') or args.img_path.endswith('.jpeg'):
            img = cv2.imread(args.img_path, cv2.IMREAD_GRAYSCALE)
    elif args.img_path.endswith('.pkl'):
            with open(args.img_path, "rb") as f:
                img = pkl.load(f)
    img=img/255
    img=np.array(img)
    img = paddle.to_tensor(img[None ,None ,: , :],dtype='float32')
    seq_prob = model(img)
    #decoder = Words(args.word_path)
    decoder = Words(params['word_path'])
    seq_prob = decoder.decode(seq_prob)
    print(f"seq_prob: {seq_prob}")
    return seq_prob



if __name__ == "__main__":
    args = get_args()
    output = main(args)

    reprod_logger = ReprodLogger()
    reprod_logger.add("output", np.array(output))
    reprod_logger.save("output_training _engine.npy")
