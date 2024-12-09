from torchvision import models
import torch.nn as nn
import torch

def ResNet18():
    # pretrained ResNet18 (have not tried more advanced versions of ResNet)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # first convolutional layer for grayscale input
    model.conv1 = nn.Conv2d(
        in_channels=1,  # Change from 3 (RGB) to 1 (grayscale)
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )
    
    # Modify the final fully connected layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # 2 classes: usable/unusable
    
    return model