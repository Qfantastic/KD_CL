import numpy as np
import subprocess
import pickle
import torch
import os

import clip
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm
# from .common import MLP, ResNet18


# cifar_path = "cifar-100-python.tar.gz"

# def unpickle(file):
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict

# cifar100_train = unpickle('data/raw/cifar-100-python/train')
# cifar100_test = unpickle('data/raw/cifar-100-python/test')

# print(cifar100_train[b'data'].shape)
# print(len(cifar100_train[b'fine_labels']))




# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# # Download the dataset
# cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# image, class_id = cifar100[3000]

# # image_np = np.asarray(image)
# # print(image_np.shape)

# image_input = preprocess(image).unsqueeze(0).to(device)
# with torch.no_grad():
#     image_features = model.encode_image(image_input)
#     print("features.shape:",image_features.shape)
#     print(image_features)
    




# net = ResNet18(10)
# print(net)
root = os.path.expanduser("~/.cache")
train = CIFAR100(root, download=True, train=True, transform=preprocess)
test = CIFAR100(root, download=True, train=False, transform=preprocess)


def get_features(dataset):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            print("images.shape:",images.shape)
            print(images[0])
            features = model.encode_image(images.to(device))
            print("features.shape:",features.shape)

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


train_features, train_labels = get_features(train)