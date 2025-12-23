import torch
import torch.nn.functional as F

import numpy as np
import pdb, os, argparse
# from scipy import misc
import time
import cv2
import imageio
from model.dark_sal_all import DarkSalNet

from utils1.data import test_dataset
from torchvision import transforms

torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
opt = parser.parse_args()

#rest()

dataset_path = '/home/dell/HJL/remote/datasets/'

model = DarkSalNet()
model.load_state_dict(torch.load('./models/DarkSalNet/DarkSalNet_ORSSD.pth.49'))

model.cuda()
model.eval()

test_datasets = ['EORSSD']

precisions = []
recalls = []
precision_list = []
recall_list = []
precision_total = []
recall_total = []
mae_total = []

thresholds = np.arange(256) / 255  # 阈值从0到255


# thresholds = [0.5]

def rgb_to_lab(rgb_img):

    rgb_img = rgb_img.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)  # [B, H, W, C]

    lab_images = []
    for img in rgb_img:

        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab_images.append(lab)


    lab_images = np.stack(lab_images, axis=0)
    lab_images = torch.tensor(lab_images).float()


    lab_images = lab_images / 255.0


    return lab_images
def calculate_precision_recall(gt, res):
    for threshold in thresholds:
        # 将预测结果二值化
        binary_res = (res > threshold).astype(int)
        binary_gt = (gt > threshold).astype(int)

        # 计算准确率和召回率
        TP = np.sum((binary_gt == 1) & (binary_res == 1))
        FP = np.sum((binary_gt == 0) & (binary_res == 1))
        FN = np.sum((binary_gt == 1) & (binary_res == 0))
        precision = TP / (TP + FP + 1e-10)
        recall = TP / (TP + FN + 1e-10)

        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls


def calculate_fbeta(precisions, recalls, beta):
    fbeta_list = []
    for precision, recall in zip(precisions, recalls):
        if precision + recall != 0:
            fbeta = (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)
            fbeta_list.append(fbeta)
        else:
            fbeta_list.append(0.0)
    return fbeta_list


def calculate_E(res, gt):
    H, W = gt.shape
    E = 0
    for x in range(H):
        for y in range(W):
            E += 1.0 / (1 + np.exp(-(np.abs(res[x, y] - gt[x, y]))))
    E /= H * W
    return E


def calculate_mae(S, GT):
    H, W = S.shape
    mae = (1 / (W * H)) * np.sum(np.abs(S - GT))
    return mae

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 转为单通道灰度
    transforms.ToTensor(),  # 转为张量并归一化到 [0, 1]
])


for dataset in test_datasets:
    save_path = './results/' + 'Dark-' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = dataset_path + dataset + '/Dark/high/test-images/'
    image2_root = dataset_path + dataset +'/test-images/'
    gt_root = dataset_path + dataset + '/test-labels/'
    test_loader = test_dataset(image_root, image2_root, gt_root, opt.testsize)
    time_sum = 0
    for i in range(test_loader.size):
        image, image2, gt, name = test_loader.load_data()


        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        image2 = image2.cuda()
        time_start = time.time()

        lab = rgb_to_lab(image)
        Pre, Pre1, Pre2, Pre3, Pre4, Pre_sig, Pre1_sig, Pre2_sig, Pre3_sig, Pre4_sig, output_rgb, all_dict = model(
            image, lab)

        time_end = time.time()
        time_sum = time_sum + (time_end - time_start)
        res = F.interpolate(output_rgb, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = np.transpose(res, (1,2,0))
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)



        res = (res * 255).astype(np.uint8)
        imageio.imsave(save_path + name, res)
        if i == test_loader.size - 1:
            print('Running time {:.5f}'.format(time_sum / test_loader.size))
            print('Average speed: {:.4f} fps'.format(test_loader.size / time_sum))

    
