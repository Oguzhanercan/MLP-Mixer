import torch
from torch import nn
from datetime import date
import numpy as np
from tqdm import tqdm

def train(args, trainloader, validloader, model, optimizer, criterion):
    device = args.device
    valid_loss_min = np.Inf
    for i in range(args.epochs):
        print("Epoch - {} Started".format(i+1))

        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for data, target in tqdm(trainloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)

        model.eval()
        for data, target in validloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item()*data.size(0)

        train_loss = train_loss/len(trainloader.sampler)
        valid_loss = valid_loss/len(validloader.sampler)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            i+1, train_loss, valid_loss))

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            if args.save == True:
                torch.save(model.state_dict(), 'model.pt')
            valid_loss_min = valid_loss
