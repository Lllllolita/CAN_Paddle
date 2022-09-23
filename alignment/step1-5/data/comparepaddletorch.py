import torch
from can_paddle.models.can import CAN as CAN_torch
from can_ref.models.can import CAN as CAN_paddle
from fnmatch import fnmatch
import paddle
import os
import numpy as np



def torch2paddle(torch_path, paddle_path):
    if not os.path.exists(torch_path):
        print("there is no model")
        return
    torch_state_dict = torch.load(torch_path, map_location=torch.device('cpu'))
    torch_state_dict = torch_state_dict['model']
    paddle_state_dict = paddle.load(paddle_path)

    linear_names = ['alpha_convert', 'word_convert']
    for k in torch_state_dict:
        # 首先对key里面bn层的参数进行转换，去掉多余的参数
        if "num_batches_tracked" in k:
            continue
        if ("conv") not in k and ("trans_layer") not in k:
            if k.endswith(".weight"):
                v1 = torch_state_dict[k].detach().cpu().numpy()
                v2 = paddle_state_dict[k].detach().numpy()
                if (not k.startswith("encoder")) and k.endswith("weight"):
                    # k1 = k[:-strip_len]
                    # 首先转置后缀为weight的权重
                    if k.endswith('weight') and ("decoder.embedding") not in k:
                        # print(k1)
                        # print(v.shape)
                        v1= np.transpose(v1)
                        # print(v.shape)
                        print(k)
                        print(np.equal(v1,v2))

            elif k.endswith(linear_names[0]) or k.endswith(linear_names[1]):
                v1 = torch_state_dict[k].detach().cpu().numpy()
                v2 = paddle_state_dict[k].detach().numpy()
                # print(k1)
                # print(v.shape)
                v1 = np.transpose(v1)
                # print(v.shape)
                print(np.equal(v1, v2))

            elif fnmatch(k, '*.fc*'):
                v1 = torch_state_dict[k].detach().cpu().numpy()
                v2 = paddle_state_dict[k].detach().numpy()
                # print(k1)
                # print(v.shape)
                v1 = np.transpose(v1)
                print(np.equal(v1, v2))
                # print(v.shape)
        # strip_len = len(".weight")
        # 然后对linear层进行处理
        elif k.endswith(".running_var"):
            k1 = k.replace("running_var", "_variance")
            v1 = torch_state_dict[k].detach().cpu().numpy()
            v2 = paddle_state_dict[k1].detach().numpy()
            print(np.equal(v1, v2))
        elif k.endswith(".running_mean"):
            k1 = k.replace("running_mean", "_mean")
            v1 = torch_state_dict[k].detach().cpu().numpy()
            v2 = paddle_state_dict[k1].detach().numpy()
            print(np.equal(v1, v2))
        else:
            v1 = torch_state_dict[k].detach().cpu().numpy()
            v2 = paddle_state_dict[k].detach().numpy()
            print(k)
            print(np.equal(v1, v2))
        # paddle_state_dict[k] = v
        # paddle.save(paddle_state_dict, paddle_path)

if __name__ == "__main__":

    paddle_path = 'can_148.pdparams'
    model_path = 'can_148.pth'

    torch2paddle(model_path, paddle_path)
