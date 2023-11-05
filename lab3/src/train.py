import torch 
import torchvision
from torch import nn
from torchvision.datasets import CIFAR10, CIFAR100
import argparse
import os
from PIL import Image
import torch
import torchvision.transforms as transforms 
from torchvision.utils import save_image
import vgg as model
# import custom_dataset as dataset
import matplotlib.pyplot as plt
import datetime

def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

learning_rate = 0.0001
learning_rate_decay = 0.00001

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, required=True,
                    help='Directory path to a batch of content images')

parser.add_argument('-c', type=str, required=True, help='number of classes')
parser.add_argument('-e', type=int, default=20, help='number of epochs')
parser.add_argument('-b', type=int, default=20, help='batch size')
parser.add_argument('-s', type=str, required=True, help='decoder weight file')
parser.add_argument('-p', type=str, required=True, help='decoder png file')
parser.add_argument('-cuda', type=str, help='[y/N]')
opt = parser.parse_args()

dataset = opt.dataset
num_epochs = opt.e
batch_size = opt.b
model_file = opt.s
model_png = opt.p
n_classes = opt.c
use_cuda = False
if opt.cuda == 'y' or opt.cuda == 'Y':
    use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

vgg = model.VGG(num_classes=n_classes)

# training encoder and new frontend
vgg.train().to(device=device)

trainset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
testset = CIFAR10(root='./data', train=False, download=True, transform=train_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

def train(num_epochs, optimizer, model, loss_fn, train_loader, scheduler, device):
    losses_train = []
    batch_loop_length = int(len(trainset) / batch_size)

    for epoch in range(1, num_epochs + 1):
        print('Epoch: ', epoch)
        loss_train = 0.0
        for imgs in range(batch_loop_length):
            imgs = next(iter(train_loader)).to(device)
            outputs = model(imgs).to(device)
            loss = loss_fn(outputs, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        scheduler.step()

        losses_train.append(loss_train / len(trainloader))

        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train/len(train_loader)))
    
    plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(losses_train[-100:])

loss_f = nn.MSELoss()
optimizer = torch.optim.Adam(vgg.parameters(), lr=learning_rate, weight_decay=learning_rate_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
train(num_epochs, optimizer, vgg, loss_f, trainloader, scheduler, device)
