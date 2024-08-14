import torch
from SoftPool import SoftPool2d

from module.resnet50 import resnet50
import torch.nn as nn
import torch.nn.functional as F
from fightingcv_attention.attention.CBAM import CBAMBlock
from module.deform_conv_v2 import DeformConv2d


# -----------------------------------------#
#   EASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
# -----------------------------------------#
class EASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(EASPP, self).__init__()
        # 1×1卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # 3×3空洞卷积，d=6
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # 3×3空洞卷积，d=12
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # 3×3空洞卷积，d=18
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # 1×1卷积，替代池化层操作，（卷积层可以进行一次选择，选择丢弃哪些信息且参数可训练）
        self.branch5_pool = SoftPool2d(kernel_size=1, stride=1)
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        # 对五个分支的结果concat（降低通道，特征融合）
        self.conv_cat = nn.Sequential(
            # DeformConv2d(dim_out*5, dim_out, 1, 0, 1, True),
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        #   torch.mean()求平均，对应第五分支Avgpool，指定维度为2,3即H,W，并保证维度不发生改变
        # -----------------------------------------#
        # global_feature = torch.mean(x, 2, True)
        # global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_pool(x)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        # -----------------------------------------#
        # 对global_feature输入进行双线性上采样，保证最终的输出结果为输入ASPP的大小，并输出；
        # align_corners设置为True，输入和输出张量由其角像素的中心点对齐，从而保留角像素处的值。
        # -----------------------------------------#
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        # -----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result



class ST_R50_connect (nn.Module):
    def __init__(self, pretrained=True, channel_1=2048, downsample_factor=16):
        super(ST_R50_connect, self).__init__()

        # ----------------------------------#
        #   对编码器进行定义
        # ----------------------------------#
        self.backbone_1 = resnet50(pretrained=pretrained)

        # -----------------------------------------#
        #   EASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        # -----------------------------------------#
        self.easpp = EASPP(dim_in=channel_1, dim_out=256, rate=16 // downsample_factor)
        self.substitute_easpp = nn.Sequential(
            nn.Conv2d(in_channels=channel_1, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.cls_conv_1 = nn.Conv2d(256, 64, 1, stride=1)

        # ---------------------------------------------------#
        # 利用CBAM进一步提取全局特征并改变通道数
        # ---------------------------------------------------#
        self.cabm = CBAMBlock(channel=256, reduction=16, kernel_size=17)

    def forward(self, x):
        # -----------------------------------------#
        #   获取CNN分支编码器提取结果，放入ASPP
        # -----------------------------------------#
        lowfeature, x_1 = self.backbone_1(x)
        # lowfeature, x_1, x1, x2, x3 = self.backbone_1(x)
        x_1 = self.easpp(x_1)
        # x_1 = self.substitute_easpp(x_1)
        x_1 = self.cabm(x_1)
        x_1 = F.interpolate(x_1, size=(lowfeature.size(2)//4, lowfeature.size(3)//4),
                            mode='bilinear', align_corners=True)
        x_1 = self.cls_conv_1(x_1)

        return x_1, lowfeature
        # return x_1, lowfeature, x1, x2, x3


class REAC_Net(nn.Module):
    def __init__(self, num_classes=2, low_level_channels = 128, low_channels_1=256, low_channels_2=512, low_channels_3=1024):
        super(REAC_Net, self).__init__()
        # ---------------------------------------------------#
        #   调用ST_XP_connect类，以获取浅层特征以及融合后的深层特征
        # ---------------------------------------------------#
        self.str50con = ST_R50_connect()
        # ----------------------------------#
        #   浅层特征边1*卷积，改变通道数
        # ----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        # -------------------------------------------------------#
        #   浅层特征与融合后的深层特征concat
        #   融合后的特征3*卷积
        # -------------------------------------------------------#
        self.cat_conv = nn.Sequential(
            # DeformConv2d(48 + 64, 64, 3, padding=1, stride=1),
            nn.Conv2d(48 + 64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # DeformConv2d(64, 64, 3, padding=1, stride=1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        # ----------------------------------------------#
        #   融合后的特征上采样，改变通道为待分类的总数，输出结果
        # ----------------------------------------------#
        self.cls_conv_2 = nn.Conv2d(64, num_classes, 1, stride=1)



        # ----------------------------------#
        #   浅层特征(low_1, low_2, low_3)边1*卷积，改变通道数
        # ----------------------------------#
        self.shortcut_conv_1 = nn.Sequential(
            nn.Conv2d(low_channels_1, low_level_channels, 1),
            nn.BatchNorm2d(low_level_channels),
            nn.ReLU(inplace=True)
        )
        self.shortcut_conv_2 = nn.Sequential(
            nn.Conv2d(low_channels_2, low_level_channels, 1),
            nn.BatchNorm2d(low_level_channels),
            nn.ReLU(inplace=True)
        )
        self.shortcut_conv_3 = nn.Sequential(
            nn.Conv2d(low_channels_3, low_level_channels, 1),
            nn.BatchNorm2d(low_level_channels),
            nn.ReLU(inplace=True)
        )
        # -------------------------------------------------------#
        #   浅层特征（low_1, low_2, low_3）与融合后的深层特征concat
        #   融合后的特征3*卷积
        # -------------------------------------------------------#
        self.cat_conv_1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )


    def forward(self, x):
        # H, W = 512, 512
        H, W = x.size(2), x.size(3)

        # -----------------------------------------#
        #  获取浅层特征与深层特征
        # -----------------------------------------#
        x, lowfeature = self.str50con(x)
        # x, lowfeature, x1, x2, x3 = self.str50con(x)

# -----------------------------------------------------------------------------------------
# 跳跃连接全添加
# -----------------------------------------------------------------------------------------
#         # -----------------------------------------#
#         #   深层特征UP 2*
#         # -----------------------------------------#
#         x = F.interpolate(x, size=(lowfeature.size(2)//8, lowfeature.size(3)//8),
#                           mode='bilinear', align_corners=True)
#         # -----------------------------------------#
#         #   与low_3融合
#         # -----------------------------------------#
#         low_3 = self.shortcut_conv_3(x3)
#         x = self.cat_conv_1(torch.cat((x, low_3), dim=1))
#         # -----------------------------------------#
#         #   深层特征UP 2*
#         # -----------------------------------------#
#         x = F.interpolate(x, size=(lowfeature.size(2)//4, lowfeature.size(3)//4),
#                           mode='bilinear', align_corners=True)
#
#         # -----------------------------------------#
#         #   与low_2融合
#         # -----------------------------------------#
#         low_2 = self.shortcut_conv_2(x2)
#         x = self.cat_conv_1(torch.cat((x, low_2), dim=1))
#         # -----------------------------------------#
#         #   深层特征UP 2*
#         # -----------------------------------------#
#         x = F.interpolate(x, size=(lowfeature.size(2) // 2, lowfeature.size(3) // 2),
#                           mode='bilinear', align_corners=True)
#
#         # -----------------------------------------#
#         #   与low_1融合
#         # -----------------------------------------#
#         low_1 = self.shortcut_conv_1(x1)
#         x = self.cat_conv_1(torch.cat((x, low_1), dim=1))
#         # -----------------------------------------#
#         #   深层特征UP 2*
#         # -----------------------------------------#
#         x = F.interpolate(x, size=(lowfeature.size(2), lowfeature.size(3)),
#                           mode='bilinear', align_corners=True)
#
#         # -----------------------------------------#
#         #   与low_feature融合
#         # -----------------------------------------#
#         lowfeature = self.shortcut_conv(lowfeature)
#         x = self.cat_conv(torch.cat((x, lowfeature), dim=1))
#
#         # -----------------------------------------#
#         #   1*改变通道数,UP2*
#         # -----------------------------------------#
#         x = self.cls_conv_2(x)
#         x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
#         return x

# ---------------------------------------------------------------------------------------
# 仅添加一次跳跃连接
# ---------------------------------------------------------------------------------------
        # -----------------------------------------#
        #   深层特征UP4*
        # -----------------------------------------#
        x = F.interpolate(x, size=(lowfeature.size(2), lowfeature.size(3)),
                          mode='bilinear', align_corners=True)

        # -----------------------------------------#
        #   浅层1*1卷积
        # -----------------------------------------#
        lowfeature = self.shortcut_conv(lowfeature)

        # -----------------------------------------#
        #   深层与浅层特征concat
        # -----------------------------------------#
        x = self.cat_conv(torch.cat((x, lowfeature), dim=1))

        # -----------------------------------------#
        #   1*,UP2*
        # -----------------------------------------#
        x = self.cls_conv_2(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

if __name__ == '__main__':

    import torch
    import time
    from thop import profile

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand(2, 3, 512, 512)
    x = x.to(device)
    net = REAC_Net()
    # net = ST_R50_connect()
    net= net.to(device)
    # print(type(net(x)))
    # print(list(net.children()))

    # 确保模型处于评估模式
    model = REAC_Net().eval()
    # 将模型移动到GPU
    model = model.to('cuda')
    # 计算FLOPs，这里不使用 clever_format，以保持 flops 为数值类型
    flops, params = profile(model, (x,), verbose=False)
    # 测量延迟
    start_time = time.perf_counter()
    output = model(x)
    end_time = time.perf_counter()
    latency = end_time - start_time
    # 计算FPS
    fps = 1 / latency
    print(output.shape)
    # 打印GFLOPs，确保 flops 是数值类型
    print('Total GFLOPS: %.3f' % (flops / 1e9))
    # 打印参数数量
    print('Total params: %d' % params)
    # 打印Latency数值
    print(f'Latency for single inference: {latency:.4f} seconds')
    # 打印FPS值
    print(f'Calculated FPS: {fps:.2f} frames per second')
