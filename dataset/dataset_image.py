import os
from PIL import Image
import cv2
import torch
from torch import nn
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F
import torch.nn.functional as Fnn
import numpy as np
import random
from .joint_transforms import Compose, JointResize, RandomHorizontallyFlip, RandomRotate

def load_batch(config, sal_root, im_name, gt_name, is_randflip_h, is_randflip_v):
    img = Image.open(os.path.join(sal_root, im_name)).convert('RGB')
    img = transforms.Resize((config.input_size, config.input_size))(img)
    anno = Image.open(os.path.join(sal_root, gt_name)).convert('L')
    anno = transforms.Resize((config.input_size, config.input_size))(anno)
    if is_randflip_h > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        anno = anno.transpose(Image.FLIP_LEFT_RIGHT)
    if is_randflip_v > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        anno = anno.transpose(Image.FLIP_TOP_BOTTOM)
 
    img = transforms.ToTensor()(img)
    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

    label1 = transforms.ToTensor()(anno)
    label2 = transforms.Resize((config.input_size//2, config.input_size//2))(anno)
    label2 = transforms.ToTensor()(label2)
    label4 = transforms.Resize((config.input_size//4, config.input_size//4))(anno)
    label4 = transforms.ToTensor()(label4)
    label8 = transforms.Resize((config.input_size//8, config.input_size//8))(anno)
    label8 = transforms.ToTensor()(label8)
    label16 = transforms.Resize((config.input_size//16, config.input_size//16))(anno)
    label16 = transforms.ToTensor()(label16)
         
    labels = {'1': label1, '2': label2, '4': label4, '8': label8, '16': label16}
 
    return img, labels

def load_batch_joint_trans(config, sal_root, im_name, gt_name, joint_trans, img_trans, mask_trans):
    label = Image.open(sal_root + gt_name).convert('L')
    img = Image.open(sal_root + im_name).convert('RGB')
    img, label = joint_trans(img, label)
    img = img_trans(img)
    label1 = mask_trans(label)
    label1 = label1.ge(0.5).float()

    label2 = transforms.Resize((config.input_size//2, config.input_size//2))(label)
    label2 = mask_trans(label2)
    label2 = label2.ge(0.5).float()

    label4 = transforms.Resize((config.input_size//4, config.input_size//4))(label)
    label4 = mask_trans(label4)
    label4 = label4.ge(0.5).float()

    label8 = transforms.Resize((config.input_size//8, config.input_size//8))(label)
    label8 = mask_trans(label8)
    label8 = label8.ge(0.5).float()

    label16 = transforms.Resize((config.input_size//16, config.input_size//16))(label)
    label16 = mask_trans(label16)
    label16 = label16.ge(0.5).float()
    labels = {'1': label1, '2': label2, '4': label4, '8': label8, '16': label16}
 
    return img, labels



def load_batch_test(config, data_root, im_name):
    img = Image.open(os.path.join(data_root, im_name)).convert('RGB')
    im_size = img.size
    img = transforms.Resize((config.input_size, config.input_size))(img)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
    return img, im_size
 

class ImageDataTrain(data.Dataset):
    def __init__(self, data_root, data_list, config):
        self.sal_root = data_root
        self.sal_source = data_list
        self.config = config
        with open(self.sal_source, 'r') as f:
            self.sal_list = [x.strip() for x in f.readlines()]

        self.sal_num = len(self.sal_list)
        self.joint_transform = Compose([JointResize(self.config.input_size), RandomHorizontallyFlip(), RandomRotate(10)])
        img_transform = [transforms.ColorJitter(0.1, 0.1, 0.1)]
        self.img_transform = transforms.Compose(
            [
                *img_transform, 
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        self.mask_transform = transforms.ToTensor()
 
    def __getitem__(self, item):
        # sal data loading
        im_root = self.sal_list[item % self.sal_num].split()[0]
        gt_root = self.sal_list[item % self.sal_num].split()[1]
        currt_img = self.sal_list[item % self.sal_num].split()[2]

        im_name = os.path.join(im_root, currt_img + '.jpg') 
        gt_name = os.path.join(gt_root, currt_img + '.png')

        #if self.config.do_aug:
        #    if self.config.do_h_flip:
        #        is_randflip_h = random.random()
        #    else:
        #        is_randflip_h = 0
        #
        #    if self.config.do_v_flip:
        #        is_randflip_v = random.random()
        #
        #    else:
        #        is_randflip_v = 0
        #else:
        #    is_randflip_h = 0
        #    is_randflip_v = 0

        #image, labels = load_batch(self.config, self.sal_root, im_name, gt_name, is_randflip_h, is_randflip_v)  
        image, labels = load_batch_joint_trans(self.config, self.sal_root, im_name, gt_name, self.joint_transform, self.img_transform, self.mask_transform)  

        sample = {'image': image, \
                  'labels': labels}
        return sample

    def __len__(self):
        return self.sal_num

class ImageDataTest(data.Dataset):
    def __init__(self, data_root, data_list, config):
        self.data_root = data_root
        self.data_list = data_list
        self.config = config
        with open(self.data_list, 'r') as f:
            self.image_list = [x.strip() for x in f.readlines()]

        self.image_num = len(self.image_list)

    def __getitem__(self, item):
        image, im_size = load_batch_test(self.config, self.data_root, self.image_list[item])

        return {'image': image, \
                'name': self.image_list[item % self.image_num], \
                'size': im_size}

    def __len__(self):
        return self.image_num



def get_loader(config, mode='train', pin=False):
    shuffle = False
    if mode == 'train':

        shuffle = True
        dataset = ImageDataTrain(config.train_root, config.train_list, config)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_thread)
    else:
        dataset = ImageDataTest(config.test_root, config.test_list, config)
        data_loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=shuffle, num_workers=config.num_thread)
    return data_loader

