import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import numpy as np
from .deeplab_resnet import resnet50_locate
from .vgg import vgg16_locate

down_dim = 32

sideout_dim_vgg = [64, 128, 256, 512, 512]

scale_factors_vgg = [1, 2, 2, 2, 1]

sideout_dim_resnet = [64, 256, 512, 1024, 2048]

scale_factors_resnet = [1, 2, 2, 2, 2]

class ResnetDilated(nn.Module):
    def __init__(self, original_resnet):
        super(ResnetDilated, self).__init__()
        from functools import partial
        # take pretrained resnet, take away AvgPool and FC
        original_resnet.layer4.apply(
            partial(self._nostride_dilate, dilate=2))

        self.layer0 = nn.Sequential(*list(original_resnet.children())[:3])
        self.maxpool= nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = original_resnet.layer1
        self.layer2 = original_resnet.layer2
        self.layer3 = original_resnet.layer3
        self.layer4 = original_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(self.maxpool(layer0))
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer0, layer1, layer2, layer3, layer4

class _DCPP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(_DCPP, self).__init__()
        # down_dim = 16
        self.scale1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), nn.ReLU()
        )
        self.scale2 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, dilation=3, padding=3), nn.ReLU()
        )
        self.scale3 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, dilation=5, padding=5), nn.ReLU()
        )
        self.scale4 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, dilation=7, padding=7), nn.ReLU()
        )
        self.global_pre = nn.Conv2d(4 * out_dim, 1, kernel_size=3, padding=1)

    def forward(self, x):
        scale1      = self.scale1(x)
        scale2      = self.scale2(x)
        scale3      = self.scale3(x)
        scale4      = self.scale4(x)
        fuse        = torch.cat((scale1, scale2, scale3, scale4), 1)
        pre         = self.global_pre(fuse)
        return pre

class ARMI(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(ARMI, self).__init__()
        self.conv_fg = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, padding=1),nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=3, padding=1),nn.ReLU()
        )
        self.residual = nn.Conv2d(planes, 1, kernel_size=1)
        self.scale_factor = scale_factor

    def interp(self, x, scale_factor):
        x = F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=True)
        return x

    def forward(self, in_pre_fg, features):
        x        = torch.cat((torch.sigmoid(in_pre_fg), features), dim=1)
        conv_fg = self.conv_fg(x)
        if self.scale_factor != 2:
            residual = self.residual(conv_fg)
            pre_fg   = in_pre_fg + residual
            return pre_fg, conv_fg
        else:
            in_pre_fg_up = self.interp(in_pre_fg , scale_factor=self.scale_factor)
            conv_fg_up = self.interp(conv_fg, scale_factor=self.scale_factor)
            residual = self.residual(conv_fg_up)
            pre_fg = in_pre_fg_up + residual
            return pre_fg, conv_fg_up

class R2Net(nn.Module):
    def __init__(self, base_model_cfg, down_dim, sideout_dim, scale_factors):
        super(R2Net, self).__init__()
        # down_dim = 32
        if base_model_cfg == 'resnet':
            original_resnet = torchvision.models.resnet101(pretrained=True)
            resnet_dilated = ResnetDilated(original_resnet)
            self.layer0 = resnet_dilated.layer0
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            self.layer1 = resnet_dilated.layer1
            self.layer2 = resnet_dilated.layer2
            self.layer3 = resnet_dilated.layer3
            self.layer4 = resnet_dilated.layer4

        self.DCPP   = _DCPP(sideout_dim[4], down_dim)

        self.reduce_layer0 = nn.Sequential(
            nn.Conv2d(sideout_dim[0], down_dim, kernel_size=1),nn.ReLU()
        )
        self.reduce_layer1 = nn.Sequential(
            nn.Conv2d(sideout_dim[1], down_dim, kernel_size=1),nn.ReLU()
        )
        self.reduce_layer2 = nn.Sequential(
            nn.Conv2d(sideout_dim[2], down_dim, kernel_size=1),nn.ReLU()
        )
        self.reduce_layer3 = nn.Sequential(
            nn.Conv2d(sideout_dim[3], down_dim, kernel_size=1),nn.ReLU()
        )
        self.reduce_layer4 = nn.Sequential(
            nn.Conv2d(sideout_dim[4], down_dim, kernel_size=1),nn.ReLU()
        )
        self.ARM_0 = ARMI(down_dim + 1, down_dim, scale_factor=scale_factors[0])
        self.ARM_1 = ARMI(down_dim + down_dim + 1, down_dim, scale_factor=scale_factors[1])
        self.ARM_2 = ARMI(down_dim + down_dim + 1, down_dim, scale_factor=scale_factors[2])
        self.ARM_3 = ARMI(down_dim + down_dim + 1, down_dim, scale_factor=scale_factors[3])
        self.ARM_4 = ARMI(down_dim + down_dim + 1, down_dim, scale_factor=scale_factors[4])

        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, x, labels=None):

        layer0 = self.layer0(x)
        layer1 = self.layer1(self.maxpool(layer0))
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        global_pre = self.DCPP(layer4)

        # reduce side-out features from ResNet
        layer0_reduced = self.reduce_layer0(layer0)
        layer1_reduced = self.reduce_layer1(layer1)
        layer2_reduced = self.reduce_layer2(layer2)
        layer3_reduced = self.reduce_layer3(layer3)
        layer4_reduced = self.reduce_layer4(layer4)

        # ARM0
        features0 = layer4_reduced
        pre0, residual_feature0 = self.ARM_0(global_pre, features0)

        # ARM1
        features1 = torch.cat((layer3_reduced, residual_feature0), dim=1)
        pre1, residual_feature1 = self.ARM_1(pre0, features1)

        # ARM2
        features2 = torch.cat((layer2_reduced, residual_feature1), dim=1)
        pre2, residual_feature2 = self.ARM_2(pre1, features2)

        # ARM3
        features3 = torch.cat((layer1_reduced, residual_feature2), dim=1)
        pre3, residual_feature3 = self.ARM_3(pre2, features3)

        # ARM4
        features4 = torch.cat((layer0_reduced, residual_feature3), dim=1)
        pre4, residual_feature4 = self.ARM_4(pre3, features4)

        if labels is not None:
            return global_pre, pre0, pre1, pre2, pre3, pre4, labels
        else:
  
            return global_pre, pre0, pre1, pre2, pre3, pre4



def build_model(base_model_cfg='vgg'):
    if base_model_cfg == 'vgg': #not implemented yet
        return R2Net(base_model_cfg, down_dim, sideout_dim_vgg, scale_factors_vgg)
    elif base_model_cfg == 'resnet':
        return R2Net(base_model_cfg, down_dim, sideout_dim_resnet, scale_factors_resnet)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()
