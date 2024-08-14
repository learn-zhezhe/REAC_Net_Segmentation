"""
图像语义分割
"""
import torch.optim

from datasets.dataset import *

from utils.train_function import *

from module.REAC_Net import REAC_Net
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# --------------------------------------------------------------------------#
#   训练模型选择：UNet、FCN、DeepLabv3plus、PSPNet、STRD_Net、HighResolutionNet
#   训练模型的保存位置重新设置，在utils.train_function
# --------------------------------------------------------------------------#
net = REAC_Net()
# ------------------------------------------------------------------#
#   指定超参数，训练轮次、学习率，权重
# ------------------------------------------------------------------#
num_epochs = 50
lr = 5e-4
wd = 1e-5
# ------------------------------------------------------------------#
#   指定训练时是否使用GPU
# ------------------------------------------------------------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------------------------------------------------------#
#   定义batch_size、以及图片尺寸大小
# ------------------------------------------------------------------#
batch_size = 8
crop_size = (512, 512)

# 测试模型输出
# x = torch.rand(1, 3, 480, 480)
# print(net(x).shape)

# 读取数据集
train_iter, test_iter = load_data_rs(batch_size, crop_size)
print(train_iter,test_iter)

# 定义优化器
optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=wd)
# optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
# 开始训练
# train(net, train_iter, test_iter, ce_loss, dice_loss, focal_loss, optimizer, num_epochs, num_steps, device)
train(net, train_iter, test_iter, ce_loss, optimizer, num_epochs, device)
# plt.show()
# plt.close()