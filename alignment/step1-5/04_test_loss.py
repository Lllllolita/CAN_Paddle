import os, torch, paddle
import numpy as np
from reprod_log import ReprodLogger
from reprod_log import ReprodDiffHelper

from can_paddle.models.loss import Loss as CANLoss_paddle
from can_ref.models.loss import CAN_Loss as CANloss_torch

from utilities import load_config, build_paddle_data_pipeline, build_torch_data_pipeline, evaluate

from can_paddle.models.can import CAN as can_paddle
from can_ref.models.can import CAN as can_torch


def test_forward(params):
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

    # 加载假数据
    inputs = np.load(os.path.join(params['base_dir'], params['fake_data_path']))
    input_masks = np.load(os.path.join(params['base_dir'], params['fake_data_mask_path']))
    labels = np.load(os.path.join(params['base_dir'], params['fake_label_path']))
    label_masks = np.load(os.path.join(params['base_dir'], params['fake_label_mask_path']))

    # 配置logger
    reprod_logger = ReprodLogger()

    # 运行paddle模型
    word_probs, counting_preds, counting_pred1, counting_pred2 = paddle_model(
        paddle.to_tensor(inputs, dtype=paddle.float32), 
        paddle.to_tensor(input_masks, dtype=paddle.float32),
        paddle.to_tensor(labels, dtype=paddle.int64), 
        paddle.to_tensor(label_masks, dtype=paddle.int64))

    #计算paddleloss并保存 
    paddleLoss = CANLoss_paddle(params)
    word_average_loss_paddle, counting_loss_paddle = paddleLoss(
        paddle.to_tensor(labels, dtype=paddle.int64), 
        paddle.to_tensor(label_masks, dtype=paddle.int64), word_probs, counting_preds, counting_pred1, counting_pred2)
    loss_paddle = word_average_loss_paddle + counting_loss_paddle

    reprod_logger.add("word_average_loss", word_average_loss_paddle.cpu().detach().numpy())
    reprod_logger.add("counting_loss", counting_loss_paddle.cpu().detach().numpy())
    reprod_logger.add("loss", loss_paddle.cpu().detach().numpy())
    reprod_logger.save(os.path.join(params['result_dir'], "loss_paddle.npy"))

    # 运行torch模型
    word_probs, counting_preds, counting_pred1, counting_pred2  = torch_model(
        torch.tensor(inputs, dtype=torch.float32).to(torch_device), 
        torch.tensor(input_masks, dtype=torch.float32).to(torch_device),
        torch.tensor(labels, dtype=torch.int64).to(torch_device),
        torch.tensor(label_masks, dtype=torch.int64).to(torch_device))
    
    # 计算torchloss并保存
    torchLoss = CANloss_torch(params)
    word_average_loss_torch, counting_loss_torch = torchLoss(
        torch.tensor(labels, dtype=torch.int64).to(torch_device),
        torch.tensor(label_masks, dtype=torch.int64).to(torch_device), word_probs, counting_preds, counting_pred1, counting_pred2)
    loss_torch = word_average_loss_torch+counting_loss_torch

    reprod_logger.add("word_average_loss", word_average_loss_torch.cpu().detach().numpy())
    reprod_logger.add("counting_loss", counting_loss_torch.cpu().detach().numpy())
    reprod_logger.add("loss", loss_torch.cpu().detach().numpy())
    reprod_logger.save(os.path.join(params['result_dir'], "loss_ref.npy"))


if __name__ == "__main__":
    params = load_config('alignment/step1-5/alignment_config.yaml')
    test_forward(params)

    # load data
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info(os.path.join(params['result_dir'],"loss_ref.npy"))
    paddle_info = diff_helper.load_info(os.path.join(params['result_dir'], "loss_paddle.npy"))

    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(diff_method="mean", diff_threshold=1e-6,path=os.path.join(params['result_dir'], "log/loss_diff.log"))