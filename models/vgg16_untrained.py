from torchvision import models
import torch.nn as nn
import torch

def VGG16():
    vgg16 = models.vgg16(weights=None) 
    vgg16.features[0] = nn.Conv2d(
        in_channels=1,  # Change to 1 for grayscale
        out_channels=64, 
        kernel_size=3,
        stride=1,
        padding=1
    )
    num_features = vgg16.classifier[6].in_features # 
    vgg16.classifier[6] = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)
    )
    return vgg16

