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
cmd.add_argument("-z", type=int, default=8)
cmd.add_argument("-e", type=int, default=50)
cmd.add_argument("-b", type=int, default=2048)
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
    model.train()   # keep track of gradient for backtracking
    losses_train = []
    
    for epoch in range(1, n_epochs+1):
        print('epoch ', epoch)  
        loss_train = 0.0
        for imgs in train_loader:   # imgs is a mini batch of data
            img, _ = imgs   # apparently this is how we unpack the images, it goes images then labels
            img = img.to(device=device)     # sends image to device
            img = img.view(img.size(0), -1)     # flattens the image, no idea why or how
            outputs = model(img)       # forward propagating through the network
            loss = loss_fn(outputs, img)    # calculate our loss
            optimizer.zero_grad()   # reset optimizer gradients to zero
            loss.backward()     # calculate loss gradients
            optimizer.step()        # iterate the optimization based on loss gradients
            loss_train += loss.item()       # update value of losses

        scheduler.step(loss_train)      # update some hyperparameters for optimization

        losses_train += [loss_train/len(train_loader)]      # update value of losses

        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train/len(train_loader)))

    plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(losses_train[-100:])
    plt.show()

train_transform = transforms.Compose([transforms.ToTensor()])
train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)
dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# the device gets decided and the variable is declared here
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f'device being used: {device}\n')

model = autoencoderMLP4()
model.to(device=device)
loss_f = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

train(num_epochs, optimizer, model, loss_f, dataloader, scheduler, device)

print(torchsummary.summary(model, (1, 28*28)))

model.eval()

images = []

for i in range(10):
    images.append(train_set.data[np.random.randint(0, len(train_set))])
    
images = torch.stack(images)
images = images.view(images.size(0), -1)
images = images.type(torch.float32)
images = Variable(images).to(device)
with torch.no_grad():
    output = model(images).to(device)

output = output.cpu()
images = images.cpu()
output = output.reshape(-1, 28, 28)
images = images.reshape(-1, 28, 28)

for i in range(10):
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(images[i], cmap='gray')
    f.add_subplot(1,2, 2)
    plt.imshow(output[i], cmap='gray')
    plt.show(block=True)

normal_images = []
images_with_noise = []

# part 5 
for i in range(10):
    training_image_idx = np.random.randint(0, len(train_set))   # get a random number from 0 to the length of the training set
    normal_images.append(train_set.data[training_image_idx])    # append the normal image to this list
    images_with_noise.append(torch.rand(train_set.data[training_image_idx]))    # append the same image with added noise to this list
    
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
    f.add_subplot(1,2,1)
    plt.imshow(normal_images[i], cmap='gray')
    f.add_subplot(1,2,2)
    plt.imshow(images_with_noise[i], cmap='gray')
    f.add_subplot(1,2,3)
    plt.imshow(output[i], cmap='gray')
    plt.show(block=True)

# -z bottleneck
# -e epochs
# -b batch size
# -s parameter file/model path
# -p name of loss image

