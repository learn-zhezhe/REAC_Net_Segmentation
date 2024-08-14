import os
import time

import torch
from torch import nn
from utils.Time import *
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.functional as F


size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)

# 创建两个SummaryWriter
train_writer = SummaryWriter("logs_seg/train")
test_writer = SummaryWriter("logs_seg/test")

# 计算预测正确的像素个数
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
        # time.sleep(0.01)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))

# 定义Accumulator类，对n个变量求和
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 使用GPU在数据集上计算模型的精度
def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()  # 模型进入验证模式
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), size(y))
            # time.sleep(0.01)
    return metric[0] / metric[1]


# 定义diceloss损失
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)#12, 6, 256, 256
        target = self._one_hot_encoder(target)#[12, 6, 256, 256]
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


# 定义Focal Loss损失函数
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction=self.reduction)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss


# 定义交叉熵损失函数
def ce_loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

# 调用FocalLoss损失
focal_loss = FocalLoss(alpha=0.25, gamma=2, reduction='mean')

# 调用DiceLoss损失
dice_loss = DiceLoss(n_classes=2)

# def train_batch(net, X, y, ce_loss, dice_loss, focal_loss, optimizer, device):
def train_batch(net, X, y, ce_loss, optimizer, device):
    X = X.to(device)
    y = y.to(device)
    # time.sleep(0.01)
    net.train()
    optimizer.zero_grad()
    pred = net(X)
    # 定义损失
    loss_ce = ce_loss(pred, y)
    loss_focal = focal_loss(pred, y)
    loss_dice = dice_loss(pred, y, softmax=True)
    # loss = 0.5 * loss_ce + 0.5 * loss_dice
    loss = 0.2 * loss_focal + 0.8 * loss_dice
    # loss = 0.5 * loss_focal + 0.5 * loss_dice
    # loss = loss_focal
    # loss = loss_dice
    # loss = loss_ce
    loss.sum().backward()
    # l = loss(pred, y)
    # l.sum().backward()
    optimizer.step()
    train_loss_sum = loss.sum()
    train_acc_sum = accuracy(pred, y)
    # time.sleep(0.01)
    return train_loss_sum, train_acc_sum
def train(net, train_iter, test_iter, ce_loss, optimizer, num_epochs, device):
    # 使用GPU进行模型训练
    timer, num_batches = Timer(), len(train_iter)
    # 将模型放在GPU上进行训练
    net = net.to(device)

    # 定义学习率余弦衰减
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=5e-5)

    for epoch in range(num_epochs):
        # 4个维度：训练损失，训练准确度，实例数，特征数
        metric = Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch(net, features, labels, ce_loss, optimizer, device)
            metric.add(l, acc, labels.shape[0], labels.numel())
            optimizer.step()
            # time.sleep(0.01)
            timer.stop()
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        # 学习率lr更新
        scheduler.step()
        # 在训练过程中记录学习率，损失和准确率
        train_writer.add_scalar("train_loss", metric[0] / metric[2], epoch)
        train_writer.add_scalar("lr_2", optimizer.param_groups[0]['lr'], epoch)
        train_writer.add_scalar("acc", metric[1] / metric[3], epoch)
        # 在测试过程中记录准确率
        test_writer.add_scalar("acc", test_acc, epoch)

        # 保存模型
        # if step % 1 == 0:
        module = net
        folder_path = 'save_model/no_ea'
        os.makedirs(folder_path, exist_ok=True)
        file_name = 'seg_072101_{}.pth'.format(epoch + 1)
        file_path = folder_path + '/' +file_name
        # 模型储存方式一 ->> 完整模型储存
        torch.save(module, file_path)
        print(f"第{epoch+1}模型已保存")
        print(f'loss {metric[0] / metric[2]:.3f}, train acc '
              f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(device)}')
    print(f'time {timer.sum()}sec')

    # writer.close()
    # 关闭writers
    train_writer.close()
    test_writer.close()
