import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import torch.nn.functional as F


class autoencoderMLP(nn.Module):
    def __init__(self, N_input=784, N_bottleneck=8, N_output=784):
        super(autoencoderMLP, self).__init__()
        N2 = 392
        self.fc1 = nn.Linear(N_input, N2)
        self.fc2 = nn.Linear(N2, N_bottleneck)
        self.fc3 = nn.Linear(N_bottleneck, N2)
        self.fc4 = nn.Linear(N2, N_output)
        self.type = "MLP4"
        self.input_shape = (1, 28*28)
        
        def forward(self, X):
            X = self.fc1(X)
            X = F.relu(X)
            X = self.fc2(X)

            X = self.fc3(X)
            X = F.relu(X)
            X = self.fc4(X)
            X = torch.sigmoid(X)

            return X
        


