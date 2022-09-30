# import sys
# sys.path.append("..")

import os, time, random, datetime, argparse
import paddle
import numpy as np
from paddle.regularizer import L2Decay
from paddlevision.optimizer import *
from tensorboardX import SummaryWriter
from utils.util import save_checkpoint, load_checkpoint, load_config

from paddlevision.datasets import get_crohme_dataset
from models.can import CAN
from models.loss import Loss as CAN_Loss
from eval import train, evaluate

parser = argparse.ArgumentParser(description='model training')
parser.add_argument('--dataset', default='CROHME', type=str, help='dataset name')
parser.add_argument('--check', action='store_true', help='if trained')
parser.add_argument('--outdir', default='outdir', action='store_true', help='save output')
parser.add_argument('--device', default='gpu:0', type=str, help='device')
parser.add_argument('--test-only', action='store_true', help='only evaluate')
args = parser.parse_args()

if not args.dataset:
    print('Please provide dataset name.')
    exit(-1)

if args.dataset == 'CROHME':
    config_file = "config.yaml"

# 加载config文件
params = load_config(config_file)

# 设置全局随机种子
random.seed(params['seed'])
np.random.seed(params['seed'])
paddle.seed(params['seed'])

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
params['device'] = args.device
# 设置计算设备(GPU\CPU)
try:
    paddle.device.set_device(params['device'])
except:
    print("device set error, use default device...")

# 多卡分布式训练，需查看文档
if paddle.distributed.get_world_size() > 1:
    paddle.distributed.init_parallel_env()

# 导入数据集
if args.dataset == 'CROHME':
    print("Loading data")
    train_loader, eval_loader = get_crohme_dataset(params)

# 导入模型
print("Creating model")

model = CAN(params)
now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
model.name = f'{params["experiment"]}_{now}_decoder-{params["decoder"]["net"]}'
print(model.name)

if args.check:
    writer = None
else:
    print("init tensorboard")
    writer = SummaryWriter(f'{params["log_dir"]}/{model.name}')

# 学习率对象，是否调参
# lr_scheduler = build_lr_scheduler(0.001, len(train_loader), 200, int(params["epochs"]), warmup_epoch=1)
lr_scheduler = build_lr_scheduler(float(params["lr"]), len(train_loader), 200, int(params["epochs"]), warmup_epoch=1)

# 梯度裁剪对象（对应torch.nn.utils.clip_grad_norm_）
if params['gradient_clip']:
    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=params['gradient'])
# 优化器对象(adamdelta)，输入统一小写
opt_name = params['optimizer'].lower()
if opt_name == 'adadelta':
    optimizer = paddle.optimizer.Adadelta(
        learning_rate=lr_scheduler,
        epsilon= float(params['eps']),
        rho=0.95,
        parameters=model.parameters(),
        weight_decay=float(params['weight_decay']),
        grad_clip=clip)
elif opt_name == 'adam':
    optimizer = paddle.optimizer.Adam(
        learning_rate=lr_scheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        parameters=model.parameters(),
        weight_decay=float(params['weight_decay']),
        grad_clip=clip)
elif opt_name == 'adamw':
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        parameters=model.parameters(),
        weight_decay=float(params['weight_decay']),
        grad_clip=clip)
elif opt_name == 'sgd':
    optimizer = paddle.optimizer.Momentum(
        learning_rate=lr_scheduler,
        momentum=0.9,
        parameters=model.parameters(),
        weight_decay=float(params['weight_decay']),
        grad_clip=clip)

# 定义loss对象，需修改
criterion = CAN_Loss(params)

# 加载预训练模型
if params['finetune']:
    print('loading from pretrain')
    print(f'pretrain path: {args.checkpoint}')
    load_checkpoint(model, optimizer, params['checkpoint'])

# 若还没开始训练过
if not args.check:
    if not os.path.exists(os.path.join(params['checkpoint_dir'], model.name)):
        os.makedirs(os.path.join(params['checkpoint_dir'], model.name), exist_ok=True)
    os.system(f'cp {config_file} {os.path.join(params["checkpoint_dir"], model.name, model.name)}.yaml')

# 多卡分布式训练
if paddle.distributed.get_world_size() > 1:
    model = paddle.DataParallel(model)

# 进入评估模式
if args.test_only and paddle.distributed.get_rank() == 0:
    load_checkpoint(model, None, params['checkpoint'])
    eval_loss, eval_word_score, eval_exprate = evaluate(model, criterion, 0, eval_loader, writer=writer)
    print(f'Epoch: 0 loss: {eval_loss:.4f} word score: {eval_word_score:.4f} ExpRate: {eval_exprate:.4f}')
    exit(0)

# 在CROHME数据集上训练
if args.dataset == 'CROHME':
    # 初始最小值，初始epoch
    min_score, init_epoch = 0, 0
    # 开始训练
    start_time = time.time()
    for epoch in range(init_epoch, params['epochs']):
        print("Start training")
        train_loss, train_word_score, train_exprate = train(model, criterion, optimizer, lr_scheduler, epoch, train_loader, writer = writer)
        if paddle.distributed.get_rank() == 0:
            if epoch >= params['valid_start']:
                print("Start evaluating")
                eval_loss, eval_word_score, eval_exprate = evaluate(model, criterion, epoch, eval_loader, writer=writer)
                print(f'Epoch: {epoch+1} loss: {eval_loss:.4f} word score: {eval_word_score:.4f} ExpRate: {eval_exprate:.4f}')

                if eval_exprate > min_score and not args.check and epoch >= params['save_start']:
                    min_score = eval_exprate
                    save_checkpoint(model, optimizer, eval_word_score, eval_exprate, epoch+1, params['checkpoint_dir'])

    # 计算总共训练时间，返回最优模型
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
