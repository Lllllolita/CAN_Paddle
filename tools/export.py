import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.static import InputSpec
import os
import sys
# import numpy as np
from models.infer_deploy import Inference
from utils.util import load_config
paddle.set_device("cpu")

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))




def get_args(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(
        description='PaddlePaddle Classification Training', add_help=add_help)

    # parser.add_argument('--model', default='CAN', help='model')
    parser.add_argument('--device', default='cpu', help='device')
    parser.add_argument('--img_size', default=None, help='image size to export')
    parser.add_argument(
        '--save-inference-dir', default='./test_model', help='path where to save')
    parser.add_argument('--pretrained', default='./test_model/predict.pdparams', help='pretrained model')
    parser.add_argument('--config_file', default='./config.yaml', help='config_file')
    parser.add_argument('--if_fast', default=True, help='if_fast')
    args = parser.parse_args()
    return args


def export(args):
    params=load_config(args.config_file)
    params['if_fast']=args.if_fast
    params['device'] = args.device
    model=Inference(params)
    layer_state_dict=paddle.load(args.pretrained)
    model.set_state_dict(layer_state_dict)
    # model.set_state_dict(layer_state_dict['model'])
    # input_spec=[InputSpec(shape=[1, 1, args.img_size, args.img_size], dtype='float32')]
    model = paddle.jit.to_static(
        model,
        input_spec=[
            InputSpec(
                shape=[1, 1, args.img_size, args.img_size], dtype='float32')
        ])
    paddle.jit.save(model, os.path.join(args.save_inference_dir, "inference"))
    print(f"inference model has been saved into {args.save_inference_dir}")


if __name__ == "__main__":
    args = get_args()
    export(args)
