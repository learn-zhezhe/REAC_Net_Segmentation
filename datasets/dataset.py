import os

import torch
import torchvision


# 读取需要进行的训练和验证的所有图像
def read_rs_images(rs_dir, is_train=True):
    # ------------------------------------------------------------------#
    #   指定进入训练以及训练验证的数据集
    # ------------------------------------------------------------------#
    txt_fname = os.path.join(rs_dir,'train.txt' if is_train else 'val.txt')

    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        # ------------------------------------------------------------------#
        #   指定与.txt文件相对应的数据集的文件夹地址
        # ------------------------------------------------------------------#

        features.append(torchvision.io.read_image(os.path.join(rs_dir, 'img', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(rs_dir, 'png', f'{fname}.png'), mode))     # 全分类时更改为label

    return features, labels


# ------------------------------------------------------------------#
#   指定图像数据相对应的，待分类的总数，以及相对应的标签的调色板
# ------------------------------------------------------------------#
# 分类标签
RS_COLORMAP = [[255, 255, 255], [0, 0, 0]]

# 定义一个函数：建立从标签RGB色彩到类索引的映射
def rs_colormap2label():
    colormap2label = torch.zeros(256**3, dtype=torch.long)
    for i, colormap in enumerate(RS_COLORMAP):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 +
                       colormap[2]] = i
    return colormap2label

# 用于加载数据的自定义数据读取类
class RSDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, crop_size, rs_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_rs_images(rs_dir, is_train=is_train)
        self.features = [
            self.normalize_image(feature)
            for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = rs_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float())

    def filter(self, imgs):
        return [
            img for img in imgs if (img.shape[1] >= self.crop_size[0] and
                                    img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = rs_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, rs_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)

# 随机剪裁遥感图像
def rs_rand_crop(feature, label, height, width):
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

# 将RS标签中的RGB值映射到其类别索引
def rs_label_indices(colormap, colormap2label):
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 +
           colormap[:, :, 2])
    return colormap2label[idx]

# 加载遥感图像数据
def load_data_rs(batch_size, crop_size):

    # 修改
    rs_dir = 'E:/我的文档/人工智能大赛数据集/wood/segmentation'
    num_workers = 0
    # num_workers = 1


    train_iter = torch.utils.data.DataLoader(
        RSDataset(True, crop_size, rs_dir), batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        RSDataset(False, crop_size, rs_dir), batch_size, drop_last=True,
        num_workers=num_workers)
    return train_iter, test_iter
