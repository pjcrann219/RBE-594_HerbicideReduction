import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_512_4(nn.Module):
    def __init__(self):
        super(CNN_512_4, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layer, input size is reduced by pooling
        # Output size is 1 for binary classification
        self.fc1 = nn.Linear(256 * 32 * 32, 1)
        
    def forward(self, x):
        # Apply convolution, activation, and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        # Flatten the tensor before the fully connected layer
        x = x.view(-1, 256 * 32 * 32)
        
        # Apply the fully connected layer with sigmoid for binary output
        x = torch.sigmoid(self.fc1(x))
        return x

class BinaryCNN(nn.Module):
    def __init__(self):
        super(BinaryCNN, self).__init__()
        
        # input = 512x512

        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # Define a pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 128)  # Adjust based on input size after convolutions
        self.fc2 = nn.Linear(128, 1)              # Output layer for binary classification

    def forward(self, x):
        # Pass through convolutional layers + ReLU + pooling
        x = self.pool(F.relu(self.conv1(x)))  # Shape: [batch_size, 32, 112, 112]
        x = self.pool(F.relu(self.conv2(x)))  # Shape: [batch_size, 64, 56, 56]
        x = self.pool(F.relu(self.conv3(x)))  # Shape: [batch_size, 128, 28, 28]
        
        # Flatten for fully connected layers
        x = x.view(-1, 128 * 28 * 28)
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Output probability for binary classification
        
        return x

# # Example usage
# model = BinaryCNN()
