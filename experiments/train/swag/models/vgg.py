"""
    VGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import math
import torch.nn as nn
import torchvision.transforms as transforms

__all__ = ['VGG16', 'VGG16BN', 'VGG16NoDrop', 'VGG16BNNoDrop', 'VGG16NoDropNoAug', 'VGG16BNNoDropNoAug',
           'VGG19', 'VGG19BN', 'VGG16Half', 'VGG16Quarter']


def make_layers(cfg, batch_norm=False):
    layers = list()
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
         512, 512, 512, 512, 'M'],
}

cfg_half = {
    16: [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M'],
}

cfg_quarter = {
    16: [16, 16, 'M', 32, 32, 'M', 64, 64, 64, 'M', 128, 128, 128, 'M', 128, 128, 128, 'M'],
}

def classifier_nodrop(num_classes):
    return nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Linear(512, num_classes),
    )

def classifier(num_classes):
    return nn.Sequential(
        nn.Dropout(),
        nn.Linear(512, 512),#cifar100
        #nn.Linear(25088,512),#places365
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Linear(512, num_classes),
    )
    
def classifier_half(num_classes):
    return nn.Sequential(
        nn.Dropout(),
        nn.Linear(256, 256),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(256, 256),
        nn.ReLU(True),
        nn.Linear(256, num_classes),
    )

def classifier_quarter(num_classes):
    return nn.Sequential(
        nn.Dropout(),
        nn.Linear(128, 128),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(128, 128),
        nn.ReLU(True),
        nn.Linear(128, num_classes),
    )

class VGG(nn.Module):
    def __init__(self, num_classes=10, depth=16, batch_norm=False,
                cfg=cfg, classifier=classifier):
        super(VGG, self).__init__()
        self.features = make_layers(cfg[depth], batch_norm)
        self.classifier = classifier(num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Base:
    base = VGG
    args = list()
    kwargs = dict()
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        #transforms.Normalize((0.4376821 , 0.4437697 , 0.47280442), (0.19803012, 0.20101562, 0.19703614))
    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #transforms.Normalize((0.45242316, 0.45249584, 0.46897713), (0.21943445, 0.22656967, 0.22850613))
    ])


class BaseNoAug:
    base = VGG
    args = list()
    kwargs = dict()
    transform_train = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


class VGG16(Base):
    pass


class VGG16NoDrop(Base):
    kwargs = {'classifier': classifier_nodrop}


class VGG16NoDropNoAug(BaseNoAug):
    kwargs = {'classifier': classifier_nodrop}


class VGG16BN(Base):
    kwargs = {'batch_norm': True}


class VGG16BNNoDrop(Base):
    kwargs = {'batch_norm': True, 'classifier': classifier_nodrop}


class VGG16BNNoDropNoAug(BaseNoAug):
    kwargs = {'batch_norm': True, 'classifier': classifier_nodrop}


class VGG19(Base):
    kwargs = {'depth': 19}


class VGG19BN(Base):
    kwargs = {'depth': 19, 'batch_norm': True}

class VGG16Half(Base):
    kwargs = {'cfg': cfg_half, 
              'classifier': classifier_half,
            }

class VGG16Quarter(Base):
    kwargs = {'cfg': cfg_quarter, 
              'classifier': classifier_quarter,
            }
