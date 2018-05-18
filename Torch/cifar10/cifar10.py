import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import torch.nn.functional as F
import torchvision.transforms as transforms

batch_size = 16
transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=1)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#------------------- training ----------------------#
epochs = 10
n_gpu = 2
is_cuda = True
model = models.resnet34(pretrained=False, num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

if n_gpu > 1:
    model = nn.DataParallel(model)
if is_cuda:
    model = model.cuda()
    criterion.cuda()

for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print('Epoch [%d/%d], Step [%d/%d], loss: %.4f' % (epoch + 1, epochs, i + 1, len(trainloader), loss.data[0]))

        # print statistics
        running_loss += loss.item()

    running_loss/=i
    print('-----------epoch: ' + str(epoch) + ', loss: ' + str(running_loss) + '-----------')

print('Finished Training')
