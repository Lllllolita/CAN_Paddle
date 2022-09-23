import torch
# from models.can import CAN as CAN_torch
# from paddle_models.can import CAN as CAN_paddle
from can_paddle.models.can import CAN as CAN_torch
from can_ref.models.can import CAN as CAN_paddle
from fnmatch import fnmatch
import paddle
import os
import numpy as np
def save_pth(params,path):
    if os.path.exists(path):
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['device'] = device
    model = CAN_torch(params)
    # state = {'model': model.state_dict()}
    torch.save(model.state_dict(),path)

def torch2paddle(torch_path,paddle_path):
    if not os.path.exists(torch_path):
        print("there is no model")
        return
    torch_state_dict = torch.load(torch_path)
    paddle_state_dict = {}
    linear_names=['alpha_convert','word_convert']
    for k in torch_state_dict['model']:
        #首先对key里面bn层的参数进行转换，去掉多余的参数
        if "num_batches_tracked" in k:
            continue
        
        strip_len=len(".weight")
        #然后对linear层进行处理
        v = torch_state_dict['model'][k]
        if (not k.startswith("encoder")) and k.endswith("weight"):
            k1=k[:-strip_len]
            #首先转置后缀为weight的权重
            if k1.endswith('weight'):
                print(k1)
                print(v.shape)
                v=np.transpose(v)
                print(v.shape)
                
            elif k1.endswith(linear_names[0]) or k1.endswith(linear_names[1]):
                print(k1)
                print(v.shape)
                v=np.transpose(v)
                print(v.shape)
                
            elif fnmatch(k1, '*.fc*'):
                print(k1)
                print(v.shape)
                v=np.transpose(v)
                print(v.shape)
        k = k.replace("running_var", "_variance")
        k = k.replace("running_mean", "_mean")
        paddle_state_dict[k] = v.cpu().detach().numpy()
        paddle.save(paddle_state_dict, paddle_path)  

def init_paddlemodel(params,paddle_path):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device="cpu"if len(paddle.device.get_available_device())==1 else paddle.device.get_available_device()
    print(device)
    params['device'] = device
    model = CAN_torch(params)
    paddle_state_dict = paddle.load(paddle_path)
    model.set_dict(paddle_state_dict)


    # return 
            # print(k1)
           
        
    # print(torch_state_dict)
if __name__=="__main__":
    params={'experiment': 'CAN', 
    'seed': 20211024, 
    'epochs': 240, 
    'batch_size': 8, 
    'workers': 0, 
    'train_parts': 1, 
    'valid_parts': 1, 
    'valid_start': 0, 
    'save_start': 0, 
    'optimizer': 'Adadelta',
    'word_num': 111,
    'lr': 1,
    'lr_decay':'cosine', 
    'step_ratio': 10, 
    'step_decay': 5, 
    'eps': '1e-6', 
    'weight_decay': '1e-4', 
    'beta': 0.9, 
    'dropout': True, 
    'dropout_ratio': 0.5, 
    'relu': True, 'gradient': 100, 
    'gradient_clip': True, 
    'use_label_mask': False, 
    'train_image_path': 'datasets/CROHME/train_images.pkl', 
    'train_label_path': 'datasets/CROHME/train_labels.txt', 
    'eval_image_path1': 'datasets/CROHME/14_test_images.pkl', 
    'eval_label_path1': 'datasets/CROHME/14_test_labels.txt', 
    'word_path': 'datasets/CROHME/words_dict.txt', 
    'collate_fn': 'collate_fn', 
    'densenet': {'ratio': 16, 'growthRate': 24, 'reduction': 0.5, 'bottleneck': True, 'use_dropout': True}, 
    'encoder': {'input_channel': 1, 'out_channel': 684}, 
    'decoder': {'net': 'AttDecoder', 'cell': 'GRU', 'input_size': 256, 'hidden_size': 256 }, 
    'counting_decoder': {'in_channel': 684, 'out_channel': 111}, 
    'attention': {'attention_dim': 512, 'word_conv_kernel': 1}, 
    'attention_map_vis_path': 'vis/attention_map', 
    'counting_map_vis_path': 'vis/counting_map', 
    'whiten_type': 'None', 
    'max_step': 256, 
    'optimizer_save': False, 
    'finetune': False, 
    'checkpoint_dir': 'checkpoints', 
    'checkpoint': '', 
    'log_dir': 'logs'}
    paddle_path="can_148.pdparams"
    model_path='can_148.pth'
    # save_pth(params,model_path)
    torch2paddle(model_path,paddle_path)
    # init_paddlemodel(params,paddle_path)