import os

import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 分类标签
RS_COLORMAP = [[255, 255, 255], [0, 0, 0]]

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

# 定义一个函数：建立从标签RGB色彩到类索引的映射
def rs_colormap2label():
    colormap2label = torch.zeros(256**3, dtype=torch.long)
    for i, colormap in enumerate(RS_COLORMAP):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 +
                       colormap[2]] = i
    return colormap2label

# 读取需要进行的训练和验证的所有图像
def read_rs_images(rs_dir, is_train=False):
    # ------------------------------------------------------------------#
    #   指定进入训练以及训练验证的数据集
    # ------------------------------------------------------------------#
    txt_fname = os.path.join(rs_dir,'test.txt' if is_train else 'test.txt')

    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        # ------------------------------------------------------------------#
        #   指定与.txt文件相对应的数据集的文件夹地址
        # ------------------------------------------------------------------#
        features.append(
            torchvision.io.read_image(
                os.path.join(rs_dir, 'img', f'{fname}.jpg')))
        labels.append(
            torchvision.io.read_image(
                os.path.join(rs_dir, 'png', f'{fname}.png'),
                mode))

    return features, labels

# 用于加载数据的自定义数据读取类
class RSDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, crop_size, rs_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_rs_images(rs_dir, is_train=False)
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

# 加载遥感图像数据
def load_data_rs(batch_size, crop_size):

    # 修改
    rs_dir = 'E:/我的文档/人工智能大赛数据集/wood/segmentation'
    num_workers = 0


    train_iter = torch.utils.data.DataLoader(
        RSDataset(True, crop_size, rs_dir), batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        RSDataset(False, crop_size, rs_dir), batch_size, drop_last=True,
        num_workers=num_workers)
    return train_iter, test_iter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size, crop_size = 2, (512, 512)
train_iter, test_iter = load_data_rs(batch_size, crop_size)

def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    # ------------------------------------------------------------------#
    #   选定相应的模型的已训练的权重，模型及权重保存位置为：module_data
    # ------------------------------------------------------------------#
    module = torch.load("save_model/reac_net/seg_072101_33.pth", map_location=device)
    pred = module(X.to(device)).argmax(dim=1)
    pred_img = pred.reshape(pred.shape[1], pred.shape[2])

    return pred_img

class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def kappa(self):
        pe_rows = np.sum(self.confusionMatrix, axis=0)
        pe_cols = np.sum(self.confusionMatrix, axis=1)
        sum_total = sum(pe_cols)
        with np.errstate(divide='ignore', invalid='ignore'):
            pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
            po = np.trace(self.confusionMatrix) / float(sum_total)
            # return (po - pe) / (1 - pe) if not np.isnan((po - pe) / (1 - pe)) else 0
            if (1 - pe) == 0:
                return None
            return (po - pe) / (1 - pe)
rs_dir = 'E:/我的文档/人工智能大赛数据集/wood/segmentation'
crop_rect = (0, 0, 512, 512)
test_images, test_labels = read_rs_images(rs_dir, False)
n = len(test_images)

def inference(num_classes):

    # 初始化性能指标
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    tn = np.zeros(num_classes)
    kappas = []
    for i in tqdm(range(n)):
        # 获得预测值
        X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
        pred = predict(X).cpu()
        # 获得标签
        colormap2label = rs_colormap2label()
        label = torchvision.transforms.functional.crop(test_labels[i], *crop_rect)
        label_indices = rs_label_indices(label, colormap2label)
        label = label_indices
        label = np.array(label)
        pred = np.array(pred)

        # 混淆矩阵以及Kappa系数计算
        metric = SegmentationMetric(num_classes)
        metric.addBatch(pred, label)
        kappa_value = metric.kappa()
        if kappa_value is not None:
            kappas.append(kappa_value)
        # kappa = metric.kappa()
        # kappas.append(kappa)

        # 初始化性能指标
        for cat in range(num_classes):
            tp[cat] += ((pred == cat) & (label == cat)).sum()
            fp[cat] += ((pred == cat) & (label != cat)).sum()
            fn[cat] += ((pred != cat) & (label == cat)).sum()
            tn[cat] += ((pred != cat) & (label != cat)).sum()
    # 计算性能指标
    iou = np.divide(tp, (tp + fp + fn))
    pre = np.divide(tp, (tp + fp))
    recall = np.divide(tp, (tp + fn))
    f1 = np.divide(2 * pre * recall, (pre + recall))
    acc = np.divide((tp + tn).sum(), (tp + fn + fp + tn).sum())
    # average_kappa = sum(kappas) / len(kappas)
    average_kappa = sum(kappas) / len(kappas) if kappas else None
    max_kappa, min_kappa = max(kappas), min(kappas)

    # 打印性能指标
    print('---------------------------------------------------')
    print('IOU:  ', iou)
    print('mIOU: ', iou.mean())
    print('pre:  ', pre)
    print('recall:  ', recall)
    print('F1', f1)
    print('Ave.F1', f1.mean())
    print('Acc', acc)
    print('---------------------------------------------------')

    return "Testing Finished!"

if __name__ == '__main__':
    num_classes = 2
    inference(num_classes)