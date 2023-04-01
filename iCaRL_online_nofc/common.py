# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d
import torch.nn.functional as F
import clip
from PIL import Image
import numpy as np

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model_clip, preprocess = clip.load('ViT-B/32', device)



def Xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self, sizes):
        super(MLP, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.conv0 = nn.Conv2d(3, 3, kernel_size=7, stride=7,
                     padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(3)
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        # self.fc = nn.Linear(nf * 8 * block.expansion, num_classes)
        self.fc = nn.Sequential(
            nn.Linear(nf * 8 * block.expansion, 512, bias=False),
            nn.ReLU(inplace=True)
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        # x = relu(self.bn0(self.conv0(x)))
        x = self.conv0(x)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # out = self.linear(out)
        # out = self.fc(out)
        return out


# def array_to_features(vx):
#     vx = vx.cpu().numpy()
#     bsize = vx.size(0)
#     all_features = []
    
#     for i in range(bsize):
#         vx_i = vx[i]
#         vx_i = np.transpose(vx_i, (1,2,0))
#         vx_i = Image.fromarray(np.uint8(vx_i*255),'RGB')
#         vx_input = preprocess(vx_i).unsqueeze(0).to(device)
#         image_features =  model_clip.encode_image(vx_input)
#         all_features.append(image_features)
#         all_features = all_features.to(device)
        
#     return torch.cat(all_features)


class CustomCLIP(nn.Module):
    def __init__(self, clip_out, ratio):
        super().__init__()
        self.ratio = ratio
        self.clip_out = clip_out
        self.adapter = Adapter(self.clip_out, 4)
        self.RestNet_adaptor = ResNet18()
        # self.fc = nn.Linear(512, num_classes)
        self.fc = nn.Linear(512, 512) 
    def forward(self, x):
        bsz = x.size(0)
        with torch.no_grad():
            # image_features = self.array_to_features(x)
            image_features = model_clip.encode_image(x.to(device))
        x_ = self.adapter(image_features)
        # x_32 = torch.reshape(bsz, (batch.shape[0], -1))
        # x_ = self.RestNet_adaptor(x.to(device))
       
        image_features = self.ratio * x_ + (1 - self.ratio) * image_features
        image_features = image_features.to(torch.float32)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # image_features = self.fc(image_features)

        
   
        return image_features
    def array_to_features(self, x):
        vx = x.cpu().numpy()
        bsize = vx.shape[0]
        all_features = []

        for i in range(bsize):
            vx_i = vx[i]
            vx_i = np.transpose(vx_i, (1,2,0))
            vx_i = Image.fromarray(np.uint8(vx_i*255),'RGB')
            vx_input = preprocess(vx_i).unsqueeze(0).to(device)
            image_features =  model_clip.encode_image(vx_input)
            all_features.append(image_features)
            
        return torch.cat(all_features).to(device).to(torch.float)

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = x.to(torch.float)
        x = self.fc(x)
        return x
    


def ResNet18(nclasses=100, nf=64):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf)

def Custom_CLIP(clip_out=512,ratio=0.2):
    

    return CustomCLIP(clip_out,ratio)