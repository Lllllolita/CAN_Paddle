import time
import paddle
from utils.util_model import Meter, cal_score
import sys
from tqdm import tqdm

def train(model,criterion,optimizer,lr_scheduler,epoch,data_loader,writer = None):
    """
    一次迭代
    """
    # 训练模式，允许梯度回传
    model.train()
    loss_meter = Meter()
    # 计算花费时间（数据读取、训练一轮）
    train_reader_cost = 0.0
    train_run_cost = 0.0
    #初始化本轮迭代加载数据量和处理批次
    total_samples = 0
    batch_past = 0
    # 精度等指标评估
    word_right = 0.0
    exp_right = 0.0
    length = 0.0
    cal_num = 0.0

    #初始化数据读取时间
    reader_start = time.time()
    # with tqdm(data_loader, total=len(data_loader)) as pbar:
    for batch_idx, (image, image_mask, label, label_mask) in enumerate(data_loader):
        image = paddle.to_tensor(image,dtype='float32')
        label = paddle.to_tensor(label, dtype='int64')
        image_mask = paddle.to_tensor(image_mask, dtype='float32')
        label_mask = paddle.to_tensor(label_mask, dtype='int64')

        train_reader_cost += time.time() - reader_start
        train_start = time.time()

        # 符号的批量和长度
        batch_num, word_length = label.shape[:2]
        #清理累积梯度
        lr_scheduler.step()
        optimizer.clear_grad()
        # 模型预测
        word_probs, counting_preds, counting_preds1, counting_preds2 = model(image, image_mask, label, label_mask)

        # 计算损失
        word_loss, counting_loss = criterion(label, label_mask, word_probs, counting_preds, counting_preds1, counting_preds2)
        loss = word_loss + counting_loss

        # 梯度回传
        loss.backward()
        optimizer.step()

        # 计算平均loss
        loss_meter.add(loss.item())
        train_run_cost += time.time() - train_start

        #评估指标,返回两个值都是float
        wordRate, ExpRate = cal_score(word_probs, label, label_mask)
        word_right = word_right + wordRate * word_length
        exp_right = exp_right + ExpRate * batch_num
        length = length + word_length
        cal_num = cal_num + batch_num

        #本轮迭代总共加载数据
        total_samples += image.shape[0]

        #本轮迭代批大小
        batch_past += 1

        if writer:
            current_step = epoch * len(data_loader) + batch_idx + 1
            writer.add_scalar('train/word_loss', word_loss.item(), current_step)
            writer.add_scalar('train/counting_loss', counting_loss.item(), current_step)
            writer.add_scalar('train/loss', loss.item(), current_step)
            writer.add_scalar('train/WordRate', wordRate, current_step)
            writer.add_scalar('train/ExpRate', ExpRate, current_step)
            writer.add_scalar('train/lr', optimizer.get_lr(), current_step)

        msg = "[Epoch {}, iter: {}] wordRate: {:.5f}, expRate: {:.5f}, lr: {:.5f}, loss: {:.5f}, avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {}, avg_ips: {:.5f} images/sec.".format(
            epoch+1, batch_idx +1, word_right / length, exp_right / cal_num,
            optimizer.get_lr(),
            loss.item(), train_reader_cost / batch_past,
            (train_reader_cost + train_run_cost) / batch_past,
            total_samples / batch_past,
            total_samples / (train_reader_cost + train_run_cost))

        # 单卡训练
        if paddle.distributed.get_rank() <= 0:
            print(msg)
            sys.stdout.flush()
            # pbar.set_description(msg)
        train_reader_cost = 0.0
        train_run_cost = 0.0
        total_samples = 0
        batch_past = 0

        reader_start = time.time()
    if writer:
        writer.add_scalar('epoch/train_loss', loss_meter.mean, epoch + 1)
        writer.add_scalar('epoch/train_WordRate', word_right / length, epoch + 1)
        writer.add_scalar('epoch/train_ExpRate', exp_right / cal_num, epoch + 1)
    return loss_meter.mean, word_right / length, exp_right / cal_num

def evaluate(model, criterion, epoch, data_loader, writer=None):
    '''
    验证模式，此时模型不会更新参数
    '''
    loss_meter = Meter()

    # 评估指标
    word_right = 0.0
    exp_right = 0.0
    length = 0.0
    cal_num = 0.0
    # with tqdm(data_loader, total=len(data_loader)) as pbar, paddle.no_grad():
    with paddle.no_grad():
        model.eval()
        for batch_idx, (image, image_mask, label, label_mask) in enumerate(data_loader):
            image = paddle.to_tensor(image,dtype='float32')
            label = paddle.to_tensor(label, dtype='int64')
            image_mask = paddle.to_tensor(image_mask, dtype='float32')
            label_mask = paddle.to_tensor(label_mask, dtype='int64')

            batch_num, word_length = label.shape[:2]
            word_probs, counting_preds, counting_preds1, counting_preds2 = model(image, image_mask, label, label_mask)
            word_loss, counting_loss = criterion(label, label_mask, word_probs, counting_preds, counting_preds1, counting_preds2)
            loss = word_loss + counting_loss
            loss_meter.add(loss.item())
            wordRate, ExpRate = cal_score(word_probs, label, label_mask)
            word_right = word_right + wordRate * word_length
            exp_right = exp_right + ExpRate * batch_num
            length = length + word_length
            cal_num = cal_num + batch_num

            if writer:
                current_step = epoch * len(data_loader) + batch_idx + 1
                writer.add_scalar('eval/word_loss', word_loss.item(), current_step)
                writer.add_scalar('eval/counting_loss', counting_loss.item(), current_step)
                writer.add_scalar('eval/loss', loss.item(), current_step)
                writer.add_scalar('eval/WordRate', wordRate, current_step)
                writer.add_scalar('eval/ExpRate', ExpRate, current_step)

            msg = "[Epoch {}, iter: {}] wordRate: {:.5f}, expRate: {:.5f}, word_loss: {:.5f}, counting_loss: {:.5f}".format(
                epoch+1, batch_idx+1, word_right / length, exp_right / cal_num, word_loss.item(), counting_loss.item())
            # 分布式训练？
            # 单卡训练
            if paddle.distributed.get_rank() <= 0:
                print(msg)
                sys.stdout.flush()
                # pbar.set_description(msg)

        if writer:
            writer.add_scalar('epoch/eval_loss', loss_meter.mean, epoch + 1)
            writer.add_scalar('epoch/eval_WordRate', word_right / length, epoch + 1)
            writer.add_scalar('epoch/eval_ExpRate', exp_right / len(data_loader.dataset), epoch + 1)

        return loss_meter.mean, word_right / length, exp_right / cal_num
