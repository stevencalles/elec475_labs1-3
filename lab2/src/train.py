import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
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

learning_rate = 0.001
learning_rate_decay = 0.0001


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
content_iter = iter(DataLoader(content_dataset, batch_size=args.b, shuffle=True))
style_iter = iter(DataLoader(style_dataset, batch_size=args.b, shuffle=True))
device = torch.device(args.cuda)
print(f"selected device for training: {device}")

total_loss = 0.0
style_loss = 0.0
content_loss = 0.0

def train(num_epochs, optimizer, model, content_loader, style_loader, device):
    print('training!')

    style_losses = []
    content_losses = []
    total_losses = []

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch: {epoch}")
        adjust_learning_rate(optimizer, iteration_count=epoch)
        total_loss = 0.0
        style_loss = 0.0
        content_loss = 0.0

        for batch in range(1, 126):
            content_imgs = next(content_loader).to(device)
            style_imgs = next(style_loader).to(device)
            loss_c, loss_s = model(content_imgs, style_imgs)
            loss_s *= gamma
            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        total_loss = (total_loss + loss) / len(content_loader)
        style_loss = (style_loss + loss_s) / len(style_loader)
        content_loss = (content_loss + loss_c) / len(content_loader)

        style_losses.append(style_loss.item())
        content_losses.append(content_loss.item())
        total_losses.append(total_loss.item())

        print('{} Epoch {}, Training loss {}, Content loss: {}, Style loss: {}'.format(datetime.datetime.now(), epoch, total_loss, content_loss, style_loss))



    plt.style.use('fivethirtyeight')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.plot(total_loss, color='r', label='loss')
    plt.plot(content_loss, color='g', label='content loss')
    plt.plot(style_loss, color='b', label='style loss')
    plt.legend()
    plt.show()
    plt.savefig(args.p)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=learning_rate_decay)
train(num_epochs=num_epochs, optimizer=optimizer, model=model, content_loader=content_iter, style_loader=style_iter, device=device)

state_dict = AdaIN_net.encoder_decoder.decoder.state_dict()
torch.save(state_dict, args.s)