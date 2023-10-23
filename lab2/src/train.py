import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
import os
from pathlib import Path
from PIL import Image
from torchvision.utils import save_image
import argparse
import AdaIN_net
import matplotlib.pyplot as plt
import datetime
from custom_dataset import custom_dataset as CustomDataset
import numpy as np

cmd = argparse.ArgumentParser()
cmd.add_argument("-content_dir", type=str, help="directory to the content images")
cmd.add_argument("-style_dir", type=str, help="directory to the style images")
cmd.add_argument("-gamma", type=float, help="value for style transfer?")
cmd.add_argument("-e", type=int, help="number of epochs")
cmd.add_argument("-b", type=int, help="batch size to use")
cmd.add_argument("-l", type=str, help="encoder model path")
cmd.add_argument("-s", type=str, help="decoder model path")
cmd.add_argument("-p", type=str, help="path to save example image")
cmd.add_argument("-cuda", type=str, default='cuda', help="determine if we want to use cuda i guess, [y/N]")
args = cmd.parse_args()

learning_rate = 0.0001
learning_rate_decay = 0.00001

def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0

class InfiniteSamplerWrapper(Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31

def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

    
def adjust_learning_rate(optimizer, iteration_count):
    lr = learning_rate / (1.0 + learning_rate_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

num_epochs = args.e
batch_size = args.b
gamma = args.gamma

encoder = AdaIN_net.encoder_decoder().encoder
decoder = AdaIN_net.encoder_decoder().decoder

encoder.load_state_dict(torch.load(args.l))
encoder = nn.Sequential(*list(encoder.children())[:31])
model = AdaIN_net.AdaIN_net(encoder, decoder)
model.train().to(args.cuda)

content_transform = train_transform()
style_transform = train_transform()

content_dataset = CustomDataset(args.content_dir, content_transform)
style_dataset = CustomDataset(args.style_dir, style_transform)
content_iter = DataLoader(content_dataset, batch_size=args.b, shuffle=True)
style_iter = DataLoader(style_dataset, batch_size=args.b, shuffle=True)
device = torch.device(args.cuda)
print(f"selected device for training: {device}")

total_loss = 0.0
style_loss = 0.0
content_loss = 0.0

def train(num_epochs, optimizer, model, content_loader, style_loader, device, scheduler):
    print('training!')

    style_losses = []
    content_losses = []
    total_losses = []

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch: {epoch}")
        total_loss = 0.0
        style_loss = 0.0
        content_loss = 0.0

        batch_total_loss = 0.0
        batch_content_loss = 0.0
        batch_style_loss = 0.0

        batch_loop_length = int(len(content_dataset) / batch_size)

        for batch in range(batch_loop_length):
            content_imgs = next(iter(content_loader)).to(device)
            style_imgs = next(iter(style_loader)).to(device)
            loss_c, loss_s = model.forward(content_imgs, style_imgs)
            loss_s *= gamma
            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_total_loss += loss
            batch_content_loss += loss_c
            batch_style_loss += loss_s

        scheduler.step()

        # adjust_learning_rate(optimizer, iteration_count=epoch)
        
        total_loss = (batch_total_loss) / batch_loop_length
        style_loss = (batch_content_loss) / batch_loop_length
        content_loss = (batch_style_loss) / batch_loop_length

        style_losses.append(style_loss.item())
        content_losses.append(content_loss.item())
        total_losses.append(total_loss.item())

        print('{} Epoch {}, Training loss {}, Content loss: {}, Style loss: {}'.format(datetime.datetime.now(), epoch, total_loss, content_loss, style_loss))
    
    # total_loss.detach()
    # content_loss.detach()
    # style_loss.detach()

    # total_loss = total_loss.cpu()
    # content_loss = content_loss.cpu()
    # style_loss = style_loss.cpu() 

    plt.style.use('fivethirtyeight')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.plot(total_losses, color='r', label='total loss')
    plt.plot(content_losses, color='g', label='content loss')
    plt.plot(style_losses, color='b', label='style loss')
    plt.legend()
    plt.grid(False)
    plt.show()
    plt.savefig(args.p)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=learning_rate_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
train(num_epochs=num_epochs, optimizer=optimizer, model=model, content_loader=content_iter, style_loader=style_iter, device=device, scheduler=scheduler)

state_dict = AdaIN_net.encoder_decoder.decoder.state_dict()
torch.save(state_dict, args.s)