import os
import sys
import random
import warnings
import re
import pickle
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.init import xavier_normal, kaiming_normal
import torchvision.datasets as dset
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
X = np.random.randn(500,3,256,256)
Y = np.random.randn(500,1)

# define dataset class
class DSB2018Data(Dataset):
    def __init__(self, X, Y, train = True):
        self.train = train
        test_size = int(X.shape[0]*0.3)
        train_size = int(X.shape[0] - test_size)
        if self.train:
            self.len = train_size
            self.X = torch.FloatTensor(X[:self.len,...])
            self.Y = torch.from_numpy(Y[:self.len,...].astype(np.float32))
        else:
            self.len = test_size
            self.X = torch.FloatTensor(X[-self.len:,...])
            self.Y = torch.from_numpy(Y[-self.len:,...].astype(np.float32))

    def __getitem__(self, index):
        x = self.X[index,...]
        y = self.Y[index,...]
        return x, y

    def __len__(self):
        return self.len


train_set = DSB2018Data(X, Y, train = True)
val_set = DSB2018Data(X, Y, train = False)
# print(train_set[0])
kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = DataLoader(train_set, batch_size=16, shuffle=True, **kwargs)
val_loader = DataLoader(val_set, batch_size=16, shuffle=True, **kwargs)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 16, 3, 1, padding=1)
        self.conv1_2 = nn.Conv2d(16, 16, 3, 1, padding=1)
        self.conv2_1 = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.conv2_2 = nn.Conv2d(32, 32, 3, 1, padding=1)

        self.conv3_1 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv3_2 = nn.Conv2d(64, 64, 3, 1, padding=1)

        self.conv4_1 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.conv4_2 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.conv5_1 = nn.Conv2d(128, 256, 3, 1, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, 3, 1, padding=1)

        self.conv6_1 = nn.Conv2d(256, 512, 3, 1, padding=1)
        self.conv6_2 = nn.Conv2d(512, 512, 3, 1, padding=1)

        self.conv7T = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv7_1 = nn.Conv2d(512, 256, 3, 1, padding=1)
        self.conv7_2 = nn.Conv2d(256, 256, 3, 1, padding=1)

        self.conv8T = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv8_1 = nn.Conv2d(256, 128, 3, 1, padding=1)
        self.conv8_2 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.conv9T = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv9_1 = nn.Conv2d(128, 64, 3, 1, padding=1)
        self.conv9_2 = nn.Conv2d(64, 64, 3, 1, padding=1)

        self.conv10T = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv10_1 = nn.Conv2d(64, 32, 3, 1, padding=1)
        self.conv10_2 = nn.Conv2d(32, 32, 3, 1, padding=1)

        self.conv11T = nn.ConvTranspose2d(32, 16, 2, 2)
        self.conv11_1 = nn.Conv2d(32, 16, 3, 1, padding=1)
        self.conv11_2 = nn.Conv2d(16, 16, 3, 1, padding=1)
        self.conv11_3 = nn.Conv2d(16, 1, 1, 1)

        self.loss_fun = nn.BCELoss()


    def forward(self, x, label):
        # normalize input data
        # x = x/255.
        # layer #1
        c1 = F.relu(self.conv1_1(x))
        c1 = F.dropout(c1, 0.1)
        c1 = F.relu(self.conv1_2(c1))
        p1 = F.max_pool2d(c1, 2)
        # layer #2
        c2 = F.relu(self.conv2_1(p1))
        c2 = F.dropout(c2, 0.1)
        c2 = F.relu(self.conv2_2(c2))
        p2 = F.max_pool2d(c2, 2)
        # layer #3
        c3 = F.relu(self.conv3_1(p2))
        c3 = F.dropout(c3, 0.1)
        c3 = F.relu(self.conv3_2(c3))
        p3 = F.max_pool2d(c3, 2)
        # layer #4
        c4 = F.relu(self.conv4_1(p3))
        c4 = F.dropout(c4, 0.1)
        c4 = F.relu(self.conv4_2(c4))
        p4 = F.max_pool2d(c4, 2)
        # layer #5
        c5 = F.relu(self.conv5_1(p4))
        c5 = F.dropout(c5, 0.1)
        c5 = F.relu(self.conv5_2(c5))
        p5 = F.max_pool2d(c5, 2)
        # layer #6
        c6 = F.relu(self.conv6_1(p5))
        c6 = F.dropout(c6, 0.1)
        c6 = F.relu(self.conv6_2(c6))

        # layer #7
        u7 = self.conv7T(c6)
        u7 = torch.cat([u7, c5], dim=1)
        c7 = F.relu(self.conv7_1(u7))
        c7 = F.dropout(c7, 0.1)
        c7 = F.relu(self.conv7_2(c7))
        # layer #8
        u8 = self.conv8T(c7)
        u8 = torch.cat([u8, c4], dim=1)
        c8 = F.relu(self.conv8_1(u8))
        c8 = F.dropout(c8, 0.1)
        c8 = F.relu(self.conv8_2(c8))
        # layer #9
        u9 = self.conv9T(c8)
        u9 = torch.cat([u9, c3], dim=1)
        c9 = F.relu(self.conv9_1(u9))
        c9 = F.dropout(c9, 0.1)
        c9 = F.relu(self.conv9_2(c9))
        # layer #10
        u10 = self.conv10T(c9)
        u10 = torch.cat([u10, c2], dim=1)
        c10 = F.relu(self.conv10_1(u10))
        c10 = F.dropout(c10, 0.1)
        c10 = F.relu(self.conv10_2(c10))
        # layer #11
        u11 = self.conv11T(c10)
        u11 = torch.cat([u11, c1], dim=1)
        c11 = F.relu(self.conv11_1(u11))
        c11 = F.dropout(c11, 0.1)
        c11 = F.relu(self.conv11_2(c11))
        output = F.sigmoid(self.conv11_3(c11))

        loss = self.loss_fun(output, label)
        return output, loss

model = UNet().cuda()
# net.apply(weights_init)
optimizer = optim.Adam(model.parameters())


for i in range(100):
    for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = Variable(data.cuda()), Variable(target.cuda())
        # optimizer.zero_grad()
        # output,loss = model(data, target)
        # loss.backward()
        # optimizer.step()
        # print('epoch: ' + str(i) + ', batch: ' + str(batch_idx) + ', loss: ' + str(loss.data[0]))
        print(batch_idx)