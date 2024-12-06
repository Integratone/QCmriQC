import torch.nn as nn
import torch

# Create 2-CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Dynamically calculate the flattened size
        flattened_size = self._calculate_flattened_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 128)  # Input size matches flattened output
        self.fc2 = nn.Linear(128, 2)  # Accept or reject
    
    def _calculate_flattened_size(self):
        # Use a dummy tensor to determine the flattened size
        dummy_input = torch.zeros(1, 1, 224, 224)  # Single grayscale image with size 224x224
        x = self.conv1(dummy_input)
        x = self.conv2(x)
        return x.numel()  # Flattened size

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = torch.nn.ReLU()(x)
        x = self.fc2(x)
        return x

