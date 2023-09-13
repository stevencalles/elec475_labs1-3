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
cmd.add_argument("-e", type=int, default=20)
cmd.add_argument("-b", type=int, default=1024)
cmd.add_argument("-s", type=argparse.FileType("w"))
cmd.add_argument("-p", type=argparse.FileType("w"))
args = cmd.parse_args()

num_epochs = args.e
batch_size = args.b
learning_rate = 1e-3

class autoencoderMLP4(nn.Module):
    def __init__(self, N_input=784, N_bottleneck=args.z, N_output=784):
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
        
def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device):
    print('training...')
    model.train()
    losses_train = []
    
    for epoch in range(1, n_epochs+1):
        print('epoch ', epoch)
        loss_train = 0.0
        for imgs in train_loader:
            img, _ = imgs
            img = img.view(img.size(0), -1)
            outputs = model(img)
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
    plt.show()

train_transform = transforms.Compose([transforms.ToTensor()])
train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)
dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

model = autoencoderMLP4()
loss_f = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
train(num_epochs, optimizer, model, loss_f, dataloader, scheduler, 'cpu')

print(summary(model, input=(1,28,28), batch_size=-1))

# model.eval()


# for imgs in dataloader:
#     img, _ = imgs
#     img = img.view(img.size(0), -1)
#     with torch.no_grad():
#         output = model(img)
#     f = plt.figure()
#     f.add_subplot(1,2,1)
#     img = img.view(img.size(0), 1, 28, 28).cpu().data
#     plt.imshow((img[2], img[3]), cmap='gray')
#     f.add_subplot(1,2,2)
#     output = output.view(output.size(0), 1, 28, 28).cpu().data
#     plt.imshow((output[2], output[3]), cmap='gray')
#     plt.show()

# -z bottleneck
# -e epochs
# -b batch size
# -s parameter file/model path
# -p name of loss image