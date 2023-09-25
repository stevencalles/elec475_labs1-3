
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
import torchsummary

cmd = argparse.ArgumentParser()
cmd.add_argument("-l", type=str, default="MLP.8.pth")
args = cmd.parse_args()

train_transform = transforms.Compose([transforms.ToTensor()])
test_set = MNIST(root='./data/mnist',  train=False, download=True, transform=train_transform)
dataloader = DataLoader(test_set, batch_size=2048, shuffle=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class autoencoderMLP4(nn.Module):
    def __init__(self, N_input=784, N_bottleneck=8, N_output=784):
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
    
model = torch.load(args.l)
model.eval()

for imgs in dataloader:
    img, _ = imgs       # grabs images, THEN labels
    img = img.view(img.size(0), -1) # flatten images
    img = Variable(img).to(device)
    with torch.no_grad():
        outputs = model(img).to(device)

# plot original images and output
plt.figure(figsize=(20, 4))
for idx in np.arange(3):
    plt.subplot(2, 10, idx+1)
    plt.imshow(img[idx].cpu().numpy().reshape(28, 28), cmap='gray')
    plt.subplot(2, 10, idx+11)
    plt.imshow(outputs[idx].cpu().detach().numpy().reshape(28, 28), cmap='gray')
plt.show()
# end of part 4

normal_images = []
images_with_noise = []

for i in range(3):
    training_image_idx = np.random.randint(0, len(dataloader.dataset))   # get a random number from 0 to the length of the training set
    current_image = dataloader.dataset[training_image_idx][0]
    normal_images.append(current_image)    # append the normal image to this list
    noise = torch.randn(current_image.size()) * 0.2     # to create noise, use randn and multiply by some number
    images_with_noise.append(current_image + noise)    # append the noisy image to this list
    
images_with_noise = torch.stack(images_with_noise)
images_with_noise = images_with_noise.view(images_with_noise.size(0), -1)
images_with_noise = images_with_noise.type(torch.float32)
images_with_noise = Variable(images_with_noise).to(device)
with torch.no_grad():
    output = model(images_with_noise).to(device)

output = output.cpu()
images_with_noise = images_with_noise.cpu()
output = output.reshape(-1, 28, 28)
images_with_noise = images_with_noise.reshape(-1, 28, 28)

# the noisy image should be at the same index in it's list as its corresponding noiseless image
for i, current_image in enumerate(images_with_noise):
    f = plt.figure()
    f.add_subplot(1,3,1)
    plt.imshow(normal_images[i].reshape(28, 28), cmap='gray')
    f.add_subplot(1,3,2)
    plt.imshow(images_with_noise[i].reshape(28, 28), cmap='gray')
    f.add_subplot(1,3,3)
    plt.imshow(output[i].reshape(28, 28), cmap='gray')
    plt.show(block=True)
# end of part 5

model.train()
images = []

for i in range(2):
    training_image_idx = np.random.randint(0, len(dataloader.dataset))   # get a random number from 0 to the length of the training set
    current_image = dataloader.dataset[training_image_idx][0]
    images.append(current_image)

images = torch.stack(images)
images = images.view(images.size(0), -1)
images = images.type(torch.float32)
images = Variable(images).to(device)

with torch.no_grad():
    bottleneck_tensors = model.encode(images).to(device)
    output = []
    for i in range(1, 9):
        weight = i / 8
        interpolated = torch.lerp(bottleneck_tensors[0], bottleneck_tensors[1], weight)
        output.append(model.decode(interpolated).to(device))

output = torch.stack(output)
output = output.view(output.size(0), -1)
output = output.type(torch.float32)
output = Variable(output).to(device)

images = images.cpu()
output = output.cpu()
images = images.reshape(-1, 28, 28)
output = output.reshape(-1, 28, 28)

f = plt.figure()
f.add_subplot(1,10,1)
plt.imshow(images[0], cmap='gray')

for i in range(2,10):
    f.add_subplot(1,10,i)
    plt.imshow(output[i-2].detach().numpy(), cmap='gray')

f.add_subplot(1,10,10)
plt.imshow(images[1], cmap='gray')

plt.show(block=True)
