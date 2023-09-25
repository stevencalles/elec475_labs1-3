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
from torch.optim import lr_scheduler
import datetime
import numpy as np
import argparse
from torchsummary import summary

cmd = argparse.ArgumentParser()
cmd.add_argument("-z", type=int, default=8)
cmd.add_argument("-e", type=int, default=50)
cmd.add_argument("-b", type=int, default=2048)
cmd.add_argument("-s", type=str, default="MLP.8.pth")
cmd.add_argument("-p", type=str, default="loss.MLP.8.png")
args = cmd.parse_args()

num_epochs = args.e
batch_size = args.b
bottleneck_size = args.z
learning_rate = 1e-3

train_transform = transforms.Compose([transforms.ToTensor()])
train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)
test_set = MNIST(root='./data/mnist',  train=False, download=True, transform=train_transform)
dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
dataloaderTest = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class autoencoderMLP4(nn.Module):
    def __init__(self, N_input=784, N_bottleneck=bottleneck_size, N_output=784):
        super(autoencoderMLP4, self).__init__()
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
    
    def encode(self, X):
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)

        return X

    def decode(self, X):
        X = self.fc3(X)
        X = F.relu(X)
        X = self.fc4(X)
        X = torch.sigmoid(X)

        return X

def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device):
    model.train().to(device)
    losses_train = []
    # Model outputs pixely images not grey scale
    # must use binary cross entropy loss 
    for epoch in range(1, n_epochs+1):
        print('epoch ', epoch)
        loss_train = 0.0
        for imgs in train_loader:
            img, _ = imgs
            img = img.view(img.size(0), -1)
            img = Variable(img).to(device)
            outputs = model(img).to(device)
            loss = loss_fn(outputs, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        scheduler.step(loss_train)

        losses_train += [loss_train/len(train_loader)]

        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train/len(train_loader)))
    plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(losses_train[-100:])
    plt.savefig(args.p)

model = autoencoderMLP4().to(device)
loss_f = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
train(num_epochs, optimizer, model, loss_f, dataloader, scheduler, device)

summary(model, (1, 28*28))

torch.save(model, args.s)