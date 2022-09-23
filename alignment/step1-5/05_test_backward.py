import paddle
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from reprod_log import ReprodLogger
from reprod_log import ReprodDiffHelper

from utilities import load_config,train_paddle, train_torch

from can_paddle.models.can import CAN as can_paddle
from can_ref.models.can import CAN as can_torch
from can_paddle.models.loss import Loss as CANLoss_paddle
from can_ref.models.loss import CAN_Loss as CANloss_torch
from paddle.regularizer import L2Decay

from paddlevision.optimizer import build_lr_scheduler
import random
import os

def test_backward(params):

    max_iter = 3
    lr = 0.0001
    cos_epochs = 200
    epochs = 240
    gradient = 80
    weight_decay = 0.0001
    # momentum = 0.9
    # lr_gamma = 0.1

    random.seed(params['seed'])
    np.random.seed(params['seed'])
    paddle.seed(params['seed'])
    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed(params['seed'])

    # set determinnistic flag
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    FLAGS_cudnn_deterministic = True

    device = "gpu"
    torch_device = torch.device("cuda:0" if device == "gpu" else "cpu")
    paddle.set_device(device)
    params['device'] = device

    # 加载paddle模型并设置为eval模式
    paddle_model = can_paddle(params)
    paddle_model.eval()
    paddle_state_dict = paddle.load(os.path.join(params['base_dir'], params['paddle_params_path']))
    paddle_model.set_dict(paddle_state_dict)


    # torch接收的设备id为cuda
    if params['device']=='gpu':
        params['device']='cuda:0'
    
    # 加载torch模型并设置为eval模式
    torch_model = can_torch(params)
    torch_model.eval()
    torch_state_dict = torch.load(os.path.join(params['base_dir'], params['torch_params_path']))
    torch_model.load_state_dict(torch_state_dict['model'])

    torch_model.to(torch_device)

    # init loss
    criterion_paddle = CANLoss_paddle(params)
    criterion_torch = CANloss_torch(params)

    # init optimizer
    lr_scheduler_paddle = build_lr_scheduler(lr, max_iter, cos_epochs, epochs, warmup_epoch=1)
    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=gradient)
    # print(lr_scheduler_paddle)
    # opt_paddle = paddle.optimizer.Adam(
    #     learning_rate=lr_scheduler_paddle,
    #     beta1=0.9,
    #     beta2=0.999,
    #     epsilon=1e-08,
    #     parameters=paddle_model.parameters(),
    #     weight_decay=L2Decay(weight_decay),
    #     grad_clip=clip)
    #
    # opt_torch = torch.optim.Adam(
    #     torch_model.parameters(),
    #     lr=lr,
    #     betas=(0.9, 0.999),
    #     eps=1e-08,
    #     weight_decay=weight_decay)
    # lr_scheduler_paddle = paddle.optimizer.lr.StepDecay(
    #     lr, step_size=max_iter // 3, gamma=lr_gamma)
    # opt_paddle = paddle.optimizer.Momentum(
    #     learning_rate=lr,
    #     momentum=momentum,
    #     parameters=paddle_model.parameters(),
    #     weight_decay=0.0,
    #     grad_clip=clip)
    #
    # opt_torch = torch.optim.SGD(torch_model.parameters(),
    #                             lr=lr,
    #                             momentum=momentum)
    opt_paddle = paddle.optimizer.Adam(
        learning_rate=lr_scheduler_paddle,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-5,
        parameters=paddle_model.parameters(),
        weight_decay= weight_decay,
        grad_clip = clip)

    opt_torch = torch.optim.Adam(
        torch_model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-5,
        weight_decay= weight_decay)
    # opt_paddle = paddle.optimizer.AdamW(learning_rate=lr,
    #                                     parameters=paddle_model.parameters(),
    #                                     beta1 = 0.9,
    #                                     beta2 = 0.999,
    #                                     epsilon=1e-08,
    #                                     weight_decay=L2Decay(weight_decay))
    # opt_torch = torch.optim.AdamW(torch_model.parameters(),
    #                               lr=lr,
    #                               betas=(0.9, 0.999),
    #                               eps=1e-08,
    #                               weight_decay= weight_decay)
    # lr_scheduler_torch = lr_scheduler.StepLR(
    #     opt_torch, step_size=max_iter // 3, gamma=lr_gamma)
    lr_scheduler_torch = lr

    # prepare logger & load data
    reprod_logger = ReprodLogger()
    
    # 加载假数据
    inputs = np.load(os.path.join(params['base_dir'], params['fake_data_path']))
    input_masks = np.load(os.path.join(params['base_dir'], params['fake_data_mask_path']))
    labels = np.load(os.path.join(params['base_dir'], params['fake_label_path']))
    label_masks = np.load(os.path.join(params['base_dir'], params['fake_label_mask_path']))

    train_paddle(paddle_model, criterion_paddle, opt_paddle, lr_scheduler_paddle, max_iter, inputs, input_masks, labels, label_masks, reprod_logger)
    train_torch(torch_model, criterion_torch, opt_torch, lr_scheduler_torch, max_iter, inputs, input_masks, labels, label_masks, gradient, reprod_logger,torch_device)

if __name__ == "__main__":
    params = load_config('alignment/step1-5/alignment_config.yaml')
    test_backward(params)

    # load data
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info(os.path.join(params['result_dir'],"losses_ref.npy"))
    paddle_info = diff_helper.load_info(os.path.join(params['result_dir'],"losses_paddle.npy"))

    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path=os.path.join(params['result_dir'],"log/backward_diff.log"),diff_threshold=1e-6)