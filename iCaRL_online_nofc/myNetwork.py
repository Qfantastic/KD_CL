import torch.nn as nn
import torch
import clip
from PIL import Image
import numpy as np
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model_clip, preprocess = clip.load('ViT-B/32', device)

class network(nn.Module):

    def __init__(self, numclass, feature_extractor):
        super(network, self).__init__()
        self.feature = feature_extractor
        # self.fc = nn.Linear(feature_extractor.fc.in_features, numclass, bias=True)
        self.fc = nn.Linear(512, numclass, bias=True)
        self.clip = model_clip

    def forward(self, input):
        
        x = self.feature(input)
        # x = self.fc(x)
        return x

    def Incremental_learning(self, numclass):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        self.fc = nn.Linear(in_feature, numclass, bias=True)
        self.fc.weight.data[:out_feature] = weight
        self.fc.bias.data[:out_feature] = bias

    def feature_extractor(self,inputs):
        return self.feature(inputs)
    
    def clip_extractor(self,inputs):
        return self.clip.encode_image(inputs.to(device))






