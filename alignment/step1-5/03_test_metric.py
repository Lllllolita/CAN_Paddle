import os, torch, paddle
from reprod_log import ReprodLogger
from reprod_log import ReprodDiffHelper

from utilities import load_config, build_paddle_data_pipeline, build_torch_data_pipeline, evaluate

from can_paddle.models.can import CAN as can_paddle
from can_ref.models.can import CAN as can_torch
from paddlevision.datasets.dataset import Words

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

    # 加载数据
    image_path = os.path.join(params['base_dir'], params['train_image_path'])
    label_path = os.path.join(params['base_dir'], params['train_label_path'])
    words = Words(os.path.join(params['base_dir'], params['word_path']))

    _, paddle_dataloader = build_paddle_data_pipeline(image_path, label_path, words)
    _, torch_dataloader = build_torch_data_pipeline(image_path, label_path, words)

     # 配置logger，加载数据
    reprod_logger = ReprodLogger()

    for idx, (paddle_batch, torch_batch) in enumerate(zip(paddle_dataloader, torch_dataloader)):
        if idx > 0:
            break
        evaluate(paddle_batch[0], paddle_batch[1], paddle_batch[2],paddle_batch[3], paddle_model, 'paddle', reprod_logger, params['result_dir'])
        evaluate(torch_batch[0].to(torch_device), torch_batch[1].to(torch_device), torch_batch[2].to(torch_device), torch_batch[3].to(torch_device), torch_model, 'ref', reprod_logger, params['result_dir'])


if __name__ == "__main__":
    params = load_config('alignment/step1-5/alignment_config.yaml')
    test_forward(params)

    # load data
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info(os.path.join(params['result_dir'],"metric_ref.npy"))
    paddle_info = diff_helper.load_info(os.path.join(params['result_dir'],"metric_paddle.npy"))

    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path=os.path.join(params['result_dir'],"log/metric_diff.log"))