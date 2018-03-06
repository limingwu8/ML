"""
UNet
Train Unet model
"""
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from dataset import get_train_valid_loader
from model import UNet
from utils import Option

def train(model, dataloader, opt, criterion, epoch):
    model.train()
    num_batches = 0
    avg_loss = 0
    with open('logs.txt', 'a') as file:
        for batch_idx, sample_batched in enumerate(dataloader):
            data = sample_batched['image']
            target = sample_batched['mask']
            data, target = Variable(data.type(opt.dtype)), Variable(target.type(opt.dtype))
            optimizer.zero_grad()
            output = model(data)
            output = (output > 0.5).type(opt.dtype)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            avg_loss += loss.data[0]
            num_batches += 1
        # avg_loss /= num_batches
        avg_loss /= len(dataloader.dataset)
        print('epoch: ' + str(epoch) + ', train loss: ' + str(avg_loss))
        file.write('epoch: ' + str(epoch) + ', train loss: ' + str(avg_loss) + '\n')

def val(model, dataloader, opt, criterion, epoch):
    model.eval()
    num_batches = 0
    avg_loss = 0
    with open('logs.txt', 'a') as file:
        for batch_idx, sample_batched in enumerate(dataloader):
            data = sample_batched['image']
            target = sample_batched['mask']
            data, target = Variable(data.type(opt.dtype)), Variable(target.type(opt.dtype))
            output = model.forward(data)
            output = (output > 0.5).type(opt.dtype)
            loss = criterion(output, target)
            avg_loss += loss.data[0]
            num_batches += 1
        # avg_loss /= num_batches
        avg_loss /= len(dataloader.dataset)

        print('epoch: ' + str(epoch) + ', validation loss: ' + str(avg_loss))
        file.write('epoch: ' + str(epoch) + ', validation loss: ' + str(avg_loss) + '\n')

def run(model, train_loader, val_loader, opt, criterion):
    for epoch in range(1, 150):
        train(model, train_loader, opt, criterion, epoch)
        val(model, val_loader, opt, criterion, epoch)

if __name__ == '__main__':
    """Train Unet model"""
    opt = Option()

    train_loader, val_loader = get_train_valid_loader(opt.root_dir, batch_size=opt.batch_size,
                                                    split=True, shuffle=opt.shuffle,
                                                    num_workers=opt.num_workers,
                                                    val_ratio=0.1, pin_memory=opt.pin_memory)

    model = UNet(input_channels=3, nclasses=1)
    if opt.n_gpu >1:
        model = nn.DataParallel(model)
    if opt.is_cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    criterion = nn.BCEWithLogitsLoss().cuda()
    # start to run
    run(model, train_loader, val_loader, opt, criterion)
