import os, glob, random
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from utils1 import transform
import numpy as np
import torch


class SalObjDataset(data.Dataset):
    def __init__(self, image_root, image_root2, gt_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.images2 = [os.path.join(image_root2, f) for f in os.listdir(image_root2) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.images2 = sorted(self.images2)
        self.gts = sorted(self.gts)
        self.size = len(self.images)


        self.img_transforms = transform.Compose([
            transform.Resize((self.trainsize,self.trainsize)),
            transform.RandRotate([0.5, 2.0], padding=[0.485, 0.456, 0.406], ignore_label=0),
            transform.RandomGaussianBlur(),
            transform.RandomHorizontalFlip(),
            transform.RandomVerticalFlip(),
            transforms.ToTensor(),
            ])

    def __getitem__(self, index):

        image = self.rgb_loader(self.images[index])
        image2 = self.rgb_loader(self.images2[index])
        gt = self.binary_loader(self.gts[index])

        image, gts = self.img_transforms(np.array(image),np.array(gt))
        image2,_ = self.img_transforms(np.array(image2),np.array(gt))
        ##gt = gt.unsqueeze(0)

        return image, image2, gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

class RestDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.rgb_loader(self.gts[index])


        image = self.img_transform(image)
        gt = self.gt_transform(gt)


        return image, gt

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __len__(self):
        return self.size

class SalObjDataset2(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.size = len(self.images)

        self.img_transforms = transform.Compose([
            transform.Resize((self.trainsize,self.trainsize)),
            transform.RandRotate([0.5, 2.0], padding=[0.485, 0.456, 0.406], ignore_label=0),
            transform.RandomGaussianBlur(),
            transform.RandomHorizontalFlip(),
            transform.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])


        return image, gt

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

class RestDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.rgb_loader(self.gts[index])


        image = self.img_transform(image)
        gt = self.gt_transform(gt)


        return image, gt

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __len__(self):
        return self.size

def get_loader(image_root, image2_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=True):

    dataset = SalObjDataset(image_root, image2_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

def get_loader2(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=True):

    dataset = RestDataset(image_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader



def get_loader3(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=True):

    dataset = SalObjDataset2(image_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, image2_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.images2 = [os.path.join(image2_root, f) for f in os.listdir(image2_root) if f.endswith('.jpg')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')]

        self.images = sorted(self.images)
        self.images2 = sorted(self.images2)
        self.gts = sorted(self.gts)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        # image1
        image = self.rgb_loader(self.images[self.index])
        t_image = self.img_transform(image).unsqueeze(0)


        image2 = self.rgb_loader(self.images2[self.index])
        t_image2 = self.img_transform(image2).unsqueeze(0)

        # ground truth
        gt = self.binary_loader(self.gts[self.index])

        # name
        name = os.path.basename(self.images[self.index])
        if name.endswith('.jpg'):
            name = name.replace('.jpg', '.png')

        self.index += 1
        return t_image, t_image2, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

