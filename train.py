import torch
import pdb, os, argparse
from datetime import datetime
from model.DarkSalNet import DarkSalNet
from utils1.data import get_loader
from utils1.func import  AvgMeter, clip_gradient, adjust_lr
import pytorch_iou
import pytorch_fm
import random
import numpy as np
from loss.losses import *
import cv2

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
set_seed(seed)


torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
# loss weights
parser.add_argument('--HVI_weight', type=float, default=1.0)
parser.add_argument('--L1_weight', type=float, default=1.0)
parser.add_argument('--D_weight',  type=float, default=0.5)
parser.add_argument('--E_weight',  type=float, default=50.0)
parser.add_argument('--P_weight',  type=float, default=1e-2)

parser.add_argument('--epoch', type=int, default=60, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')  # 原学习率为1e-4
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=40, help='every n epochs decay learning rate')
opt = parser.parse_args()

print('Learning Rate: {}'.format(opt.lr))


#rest()
# build models
model = DarkSalNet()


model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

image_root = '/home/dell/HJL/remote/datasets/ORSSD/Dark/low/train-images/'
gt_res_root = '/home/dell/HJL/remote/datasets/ORSSD/train-images/'
gt_root = '/home/dell/HJL/remote/datasets/ORSSD/train-labels/'

train_loader = get_loader(image_root, gt_res_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)


def rgb_to_lab(rgb_img):

    rgb_img = rgb_img.detach().permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)  # [B, H, W, C]

    lab_images = []
    for img in rgb_img:

        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab_images.append(lab)


    lab_images = np.stack(lab_images, axis=0)
    lab_images = torch.tensor(lab_images).float()


    lab_images = lab_images / 255.0


    return lab_images


def init_loss():
    L1_weight = opt.L1_weight
    D_weight = opt.D_weight
    E_weight = opt.E_weight
    P_weight = 1.0

    L1_loss = L1Loss(loss_weight=L1_weight, reduction='mean').cuda()
    D_loss = SSIM(weight=D_weight).cuda()
    E_loss = EdgeLoss(loss_weight=E_weight).cuda()
    P_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1, 'conv3_4': 1, 'conv4_4': 1}, perceptual_weight=P_weight,
                            criterion='mse').cuda()
    return L1_loss, P_loss, E_loss, D_loss

L1_loss,P_loss,E_loss,D_loss = init_loss()

CE = torch.nn.BCEWithLogitsLoss()
IOU = pytorch_iou.IOU(size_average=True)
floss = pytorch_fm.FLoss()

size_rates = [0.75, 1, 1.25]  # multi-scale training


def train(train_loader, model, optimizer, epoch):
    model.train()
    loss_record1, loss_record2, loss_record3, loss_record4, loss_record5, loss_record6 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            images, gt_rgb, gts = pack
            images = Variable(images).cuda()
            gt_rgb = Variable(gt_rgb).cuda()
            gt_sal = Variable(gts).cuda()

            # multi-scale training samples
            trainsize = int(round(opt.trainsize * rate / 32) * 32)

            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gt_rgb = F.interpolate(gt_rgb, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gt_sal = F.interpolate(gt_sal, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            lab = rgb_to_lab(images)
            Pre, Pre1, Pre2, Pre3, Pre4, Pre_sig, Pre1_sig, Pre2_sig, Pre3_sig, Pre4_sig, output_rgb = model(
                images, lab)

            output_uvi, gt_uvi = model.UVIT(output_rgb, rgb_to_lab(output_rgb)), model.UVIT(gt_rgb, rgb_to_lab(gt_rgb))


            loss_uvi = (L1_loss(output_uvi, gt_uvi) + D_loss(output_uvi, gt_uvi) +
                        E_loss(output_uvi, gt_uvi) + opt.P_weight * P_loss(output_uvi, gt_uvi)[0])
            loss_rgb = (L1_loss(output_rgb, gt_rgb) + D_loss(output_rgb, gt_rgb) +
                        E_loss(output_rgb, gt_rgb) + opt.P_weight * P_loss(output_rgb, gt_rgb)[0])

            loss1 = CE(Pre1, gt_sal) + IOU(Pre1_sig, gt_sal) + floss(Pre1_sig, gt_sal)
            loss2 = CE(Pre2, gt_sal) + IOU(Pre2_sig, gt_sal) + floss(Pre2_sig, gt_sal)
            loss3 = CE(Pre3, gt_sal) + IOU(Pre3_sig, gt_sal) + floss(Pre3_sig, gt_sal)
            loss4 = CE(Pre4, gt_sal) + IOU(Pre4_sig, gt_sal) + floss(Pre4_sig, gt_sal)
            loss5 = CE(Pre, gt_sal) + IOU(Pre_sig, gt_sal) + floss(Pre_sig, gt_sal)

            loss_sal = loss1 + loss2 + loss3 + loss4 + loss5

            loss = loss_rgb + opt.HVI_weight * loss_uvi + loss_sal

            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            if rate == 1:
                loss_record1.update(loss1.data, opt.batchsize)
                loss_record4.update(loss4.data, opt.batchsize)

        if i % 20 == 0 or i == total_step:
            print(
                '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Loss: {:.4f}, Loss1: {:.4f}'.
                format(datetime.now(), epoch, opt.epoch, i, total_step,
                       opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), loss.data, loss1.data))

    save_path = 'models/DarkSalNet/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch + 1) >= 30:
        torch.save(model.state_dict(), save_path + 'DarkSalNet_ORSSD_m.pth' + '.%d' % epoch)


print("Let's go!")
if __name__ == '__main__':
    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch)
