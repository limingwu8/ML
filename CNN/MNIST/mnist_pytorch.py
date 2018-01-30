import torch
import numpy as np
from PIL import Image
from torch.autograd import Variable
import torch.nn as nn
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.optim as optim
import torch

batch_size = 32
# download dataset
## load mnist dataset
root = './data'
download = False
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
test_set = dset.MNIST(root=root, train=False, transform=trans)

kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False, **kwargs)

# network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, 1)
        self.conv2 = nn.Conv2d(16, 32, 5, 1)
        self.fn1 = nn.Linear(4*4*32, 512)
        self.fn2 = nn.Linear(512, 10)
        self.loss_fun = nn.CrossEntropyLoss()
    def forward(self, x, target):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, 4*4*32)
        x = self.fn1(x)
        x = self.fn2(x)
        x = F.softmax(x)
        loss = self.loss_fun(x, target)
        return x, loss

model = Net().cuda()

optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

def train(epoch = 10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = Variable(data.cuda()), Variable(target.cuda())
        # optimizer.zero_grad()
        # output,loss = model(data, target)
        # loss.backward()
        # optimizer.step()
        print(batch_idx)
    # print('train epoch: ' + str(epoch) + ', loss: ' + str(loss.data[0]))

for epoch in range(1, 20):
    train(epoch)