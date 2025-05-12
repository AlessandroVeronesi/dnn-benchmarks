
import os
import argparse

import torch
from torch import optim
import torch.nn as nn

from dataset import getMNIST
import models
import utils

###################################################

datasetdir = os.getenv('TORCH_DATASETPATH', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data'))
# traindir   = os.getenv('TORCH_TRAINPATH', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data'))
traindir   = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

###################################################

def train(model, savefile, epochs, dataloader):

    ## Prepare Model for Training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()
    model = model.to(device)

    ## Train Settings
    learning_rate = 1e-3
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [epochs//4, epochs//2, epochs//1.25], gamma = 0.1)

    ## Train Model
    print(f'-I({__file__}): Training model...')
    utils.train_model(model, dataloader, epochs, loss_fn, optimizer, scheduler, device)
    print(f'-I({__file__}): Network trained')

    ## Save Trained Network
    model = model.to('cpu')
    torch.save(model.state_dict(), savefile)
    print(f'-I({__file__}): Weights saved into {savefile}')

###################################################

if __name__ == '__main__':

    networks = [
        'lenet'
    ]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()

    parser.add_argument('--model',      choices=networks, required=True)
    parser.add_argument('--epochs',     type=int, default=10,   help='Training epochs')
    parser.add_argument('--batchsize',  type=int, default=64,   help='Batch Size')

    args = parser.parse_args()

    ## Get Network
    print(f'-I({__file__}): Loading model...')

    if args.model=='lenet':
        model=models.LeNet()
    else:
        raise ValueError('unsupported model')

    model.to(device)
    savefile = os.path.join(traindir, f'fp32_{args.model}_mnist.pth')
    print(f'-I({__file__}): Model loaded')

    ## Dataset
    img_size = (28, 28)
    trainloader, _ = getMNIST(datasetdir, img_size, args.batchsize, device)

    train(model, savefile, args.epochs, trainloader)
