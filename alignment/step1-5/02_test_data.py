import os
import numpy as np

from reprod_log import ReprodLogger, ReprodDiffHelper

from utilities import load_config,build_paddle_data_pipeline, build_torch_data_pipeline
from paddlevision.datasets.dataset import Words

def test_data_pipeline(params):
    image_path = os.path.join(params['base_dir'], params['train_image_path'])
    label_path = os.path.join(params['base_dir'], params['train_label_path'])
    words = Words(os.path.join(params['base_dir'], params['word_path']))

    # 构造paddle和torch两个数据集
    paddle_dataset, paddle_dataloader = build_paddle_data_pipeline(image_path, label_path, words)
    torch_dataset, torch_dataloader = build_torch_data_pipeline(image_path, label_path, words)

    # logger相关
    logger_paddle_data = ReprodLogger()
    logger_torch_data = ReprodLogger()

    logger_paddle_data.add("length", np.array(len(paddle_dataset)))
    logger_torch_data.add("length", np.array(len(torch_dataset)))

    for idx, (paddle_batch, torch_batch) in enumerate(zip(paddle_dataloader, torch_dataloader)):
        if idx >= 5:
            break
        logger_paddle_data.add(f"dataloader_{idx}", paddle_batch[0].numpy())
        logger_torch_data.add(f"dataloader_{idx}", torch_batch[0].detach().cpu().numpy())
    logger_paddle_data.save(os.path.join(params['result_dir'],'result/data_paddle.npy'))
    logger_torch_data.save(os.path.join(params['result_dir'],'result/data_ref.npy'))


if __name__ == "__main__":
    params = load_config('alignment/step1-5/alignment_config.yaml')
    test_data_pipeline(params)

    # load data
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info(os.path.join(params['result_dir'],'data_ref.npy'))
    paddle_info = diff_helper.load_info(os.path.join(params['result_dir'],'data_paddle.npy'))

    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path=os.path.join(params['result_dir'],'log/data_diff.log'))