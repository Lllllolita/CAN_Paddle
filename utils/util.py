import os, yaml, paddle

def load_config(yaml_path):
    try:
        with open(yaml_path, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    except:
        print('尝试UTF-8编码....')
        with open(yaml_path, 'r', encoding='UTF-8') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    if not params['experiment']:
        print('实验名不能为空!')
        exit(-1)
    if not params['train_image_path']:
        print('训练图片路径不能为空！')
        exit(-1)
    if not params['train_label_path']:
        print('训练label路径不能为空！')
        exit(-1)
    if not params['word_path']:
        print('word dict路径不能为空！')
        exit(-1)
    if 'train_parts' not in params:
        params['train_parts'] = 1
    if 'valid_parts' not in params:
        params['valid_parts'] = 1
    if 'valid_start' not in params:
        params['valid_start'] = 0
    if 'word_conv_kernel' not in params['attention']:
        params['attention']['word_conv_kernel'] = 1
    return params

def save_checkpoint(model, optimizer, word_score, ExpRate_score, epoch, checkpoint_dir):
    model_filename = f'{os.path.join(checkpoint_dir, model.name)}/{model.name}_WordRate-{word_score:.4f}_ExpRate-{ExpRate_score:.4f}_{epoch}.pdparams'
    paddle.save(model.state_dict(),model_filename)
    if optimizer:
        opt_filename = f'{os.path.join(checkpoint_dir, model.name)}/{model.name}_WordRate-{word_score:.4f}_ExpRate-{ExpRate_score:.4f}_{epoch}.pdopt'
        paddle.save(optimizer.state_dict(),opt_filename)
    print(f'Save checkpoint: {epoch}\n')


def load_checkpoint(model, optimizer, path):
    state = paddle.load(path + '.pdparams')
    model.set_state_dict(state)
    if optimizer:
        opt_state = paddle.load(path +'.pdopt')
        optimizer.set_state_dict(opt_state)