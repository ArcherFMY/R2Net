import torch
from collections import OrderedDict
from torch.nn import utils, functional as F
from torch.optim import Adam
from torchvision import transforms
from torch.autograd import Variable
from torch.backends import cudnn
# from networks.poolnet import build_model, weights_init
from networks.r2net import build_model, weights_init
import numpy as np
import os
import torchvision.utils as vutils
import cv2
import time

class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        self.lr_decay_epoch = [15,]
        self.build_model()
        if config.mode == 'test':
            print('Loading pre-trained model from %s...' % self.config.model)
            if len(self.config.test_on_gpus) > 0:
                self.net = torch.nn.DataParallel(self.net, device_ids=self.config.test_on_gpus)
            self.net.load_state_dict(torch.load(self.config.model))
            self.net.eval()

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    # build the network
    def build_model(self):
        self.net = build_model(self.config.arch)
        if self.config.cuda:
            self.net = self.net.cuda()
        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr, weight_decay=self.wd)
        self.print_network(self.net, 'R2Net Structure')
    
    def test(self):

        mode_name = 'sal_fuse'
        if not os.path.exists(self.config.test_fold):
            os.mkdir(self.config.test_fold)
        time_s = time.time()
        img_num = len(self.test_loader)

        for i, data_batch in enumerate(self.test_loader):
            print('Testing %d/%d \n' % (i+1, img_num))
            images, name, im_size = data_batch['image'], data_batch['name'][0], data_batch['size']
            with torch.no_grad():
                global_pre, pre0, pre1, pre2, pre3, pre4 = self.net(images)
                save_salmap(self.config.test_fold, name, im_size, pre4)
                                
        time_e = time.time()
        print('Speed: %f FPS' % (img_num/(time_e-time_s)))
        print('Test Done!')

    # training phase
    def train(self):
        if self.config.batch_size>32:
            self.net.train()
        else:
            self.net.eval()

        if len(self.config.train_on_gpus)>0:
            #cudnn.benchmark = True
            self.net = torch.nn.DataParallel(self.net, device_ids=self.config.train_on_gpus)

        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        aveGrad = 0
        t_update = 0
        for epoch in range(self.config.epoch):
            r_sal_loss= 0
            self.net.zero_grad()
            time_s = time.time()
            for i, data_batch in enumerate(self.train_loader):
                image, labels = data_batch['image'], data_batch['labels']
                if (image.size(2) != labels['1'].size(2)) or (image.size(3) != labels['1'].size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                cudnn.benchmark = True
                global_pre, pre0, pre1, pre2, pre3, pre4, labels = self.net(image, labels)
                sal_loss_pre4 = F.binary_cross_entropy_with_logits(pre4, labels['1'], reduction='sum')
                sal_loss_pre3 = F.binary_cross_entropy_with_logits(pre3, labels['2'], reduction='sum')
                sal_loss_pre2 = F.binary_cross_entropy_with_logits(pre2, labels['4'], reduction='sum')
                sal_loss_pre1 = F.binary_cross_entropy_with_logits(pre1, labels['8'], reduction='sum')
                sal_loss_pre0 = F.binary_cross_entropy_with_logits(pre0, labels['16'], reduction='sum')
                sal_loss_global_pre = F.binary_cross_entropy_with_logits(global_pre, labels['16'], reduction='sum')

                sal_loss_fuse = 256*sal_loss_global_pre + 256*sal_loss_pre0 + 64*sal_loss_pre1 + 16*sal_loss_pre2 + 4*sal_loss_pre3 + sal_loss_pre4
                sal_loss = sal_loss_fuse / (self.iter_size * self.config.batch_size)
                r_sal_loss += sal_loss.data
                sal_loss.backward()

                aveGrad += 1
                # accumulate gradients as done in DSS
                if aveGrad % self.iter_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    aveGrad = 0
                if i % (self.show_every // self.config.batch_size) == 0:
                    time_e = time.time()
                    time_total = time_e - time_s
                    time_s = time.time()
                    if i == 0:
                        x_showEvery = 1
                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Sal : %10.4f || time: [%4.1f]' % (
                        epoch, self.config.epoch, i, iter_num, r_sal_loss/x_showEvery, time_total))
                    print('Learning rate: ' + str(self.lr))
                    r_sal_loss= 0

            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/models/epoch_%d.pth' % (self.config.save_folder, epoch + 1))

            if epoch in self.lr_decay_epoch:
                self.lr = self.lr * 0.1
                self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr, weight_decay=self.wd)

        torch.save(self.net.state_dict(), '%s/models/final.pth' % self.config.save_folder)

def save_salmap(path, name, size, pred):
    pred = np.squeeze(torch.sigmoid(pred).cpu().data.numpy())
    multi_fuse = 255 * pred
    multi_fuse = cv2.resize(multi_fuse, (size[0], size[1]))
    cv2.imwrite(os.path.join(path, name[:-4] + '.png'), multi_fuse)


