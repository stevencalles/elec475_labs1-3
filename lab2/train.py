import argparse
import os
from PIL import Image
import torch
import torchvision.transforms as transforms 
from torchvision.utils import save_image
import AdaIN_net as net
import custom_dataset as dataset
import matplotlib.pyplot as plt
import datetime

def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

    
parser = argparse.ArgumentParser()
parser.add_argument('-content_dir', type=str, required=True,
                    help='Directory path to a batch of content images')
parser.add_argument('-style_dir', type=str, required=True,
                    help='Directory path to a batch of style images')
parser.add_argument('-gamma', type=float, default=1.0, help='weight of content loss')
parser.add_argument('-e', type=int, default=20, help='number of epochs')
parser.add_argument('-b', type=int, default=20, help='number of batches')
parser.add_argument('-l', type=str, required=True, help='encoder weight file')
parser.add_argument('-s', type=str, required=True, help='decoder weight file')
parser.add_argument('-p', type=str, required=True, help='decoder png file')
parser.add_argument('-cuda', type=str, help='[y/N]')
opt = parser.parse_args()

content_dir = opt.content_dir
style_dir = opt.style_dir
gamma = opt.gamma
epochs = opt.e
n_batch = opt.b
encoder_file = opt.l
decoder_file = opt.s
decoder_png = opt.p
use_cuda = False
if opt.cuda == 'y' or opt.cuda == 'Y':
    use_cuda = True
out_dir = './output/'
os.makedirs(out_dir, exist_ok=True)

device = torch.device("cuda" if use_cuda else "cpu")

encoder = net.encoder_decoder.encoder
encoder.load_state_dict(torch.load(encoder_file, map_location='cpu'))

# Training the decoder
decoder = net.encoder_decoder.decoder
model = net.AdaIN_net(encoder, decoder)
model.train()
model.to(device=device)

# Load the dataset using dataload
content_dataset = dataset.custom_dataset(content_dir, transform=train_transform())
style_dataset = dataset.custom_dataset(style_dir, transform=train_transform())
content_loader = torch.utils.data.DataLoader(dataset=content_dataset, batch_size=n_batch, shuffle=True)
style_loader = torch.utils.data.DataLoader(dataset=style_dataset, batch_size=n_batch, shuffle=True)

# Optimizer
optimizer = torch.optim.Adam(model.decoder.parameters(), lr=0.001)

# Scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=gamma)

# Loss function
loss_func = torch.nn.MSELoss()
loss_epochs = []
loss_epochs_content = []
loss_epochs_style = []

# Training loop
for epoch in range(epochs):
    loss_train = 0
    loss_train_content = 0
    loss_train_style = 0

    for batch in range(n_batch) :
        content = next(iter(content_loader)).to(device)
        style = next(iter(style_loader)).to(device)      

        loss_c, loss_s = model.forward(content, style)   
        optimizer.zero_grad()
        
        loss = loss_c + (loss_s * gamma)
        loss.backward()
        optimizer.step()
       
        loss_train += loss.item()
        loss_train_content += loss_c.item()
        loss_train_style += loss_s.item()
    
    scheduler.step()
    loss_epochs.append(loss_train / n_batch)
    loss_epochs_content.append(loss_train_content / n_batch)
    loss_epochs_style.append(loss_train_style / n_batch)
    # print style, content, and total loss and have datetime
    print('Epoch [{}/{}], Content Loss: {:.4f}, Style Loss: {:.4f}, Total Loss: {:.4f}, Time: {}'.format(epoch+1, epochs, loss_train_content / n_batch, loss_train_style / n_batch, loss_train / n_batch, datetime.datetime.now().strftime("%H:%M:%S")))

plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')
# plot all three losses
plt.plot(loss_epochs_content)
plt.plot(loss_epochs_style)
plt.plot(loss_epochs)
plt.legend(['Content Loss', 'Style Loss', 'Total Loss'])
plt.savefig('loss.png')

    #print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, batch+1, n_batch, loss.item()))
        
# Save the model
torch.save(model.decoder.state_dict(), decoder_file)
print('Model saved to {}'.format(decoder_file))

# # Save the output
# content = content.to(device=device)
# style = style.to(device=device)
# model.eval()
# with torch.no_grad():
#     output = model.forward(model.encode(content), model.encode(style))
#     save_image(output.cpu(), decoder_png)
#     print('Output saved to {}'.format(decoder_png))
