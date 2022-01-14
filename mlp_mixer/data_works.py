import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
import os
from pathlib import Path
import torch


class Custom_Dataset():

    def __init__(self, directory, mode="train"):
        self.path = Path(directory)
        Path.ls = lambda x: list(x.iterdir())
        try:
            files = os.listdir(directory+"/data")
        except:
            print("wrong path")
        self.x = [torch.tensor(np.transpose(np.array(Image.open(img)), (2, 0, 1))).type(
            torch.FloatTensor) for img in (path/files[0]).ls()]
        self.x = torch.stack(self.x)/255
        self.y = torch.tensor([0]*len((path/files[0]).ls()))
        for i in range(len(files)-1):
            self.x2 = [torch.tensor(np.transpose(np.array(Image.open(img)), (2, 0, 1))).type(
                torch.FloatTensor) for img in (path/files[i+1]).ls()]
            self.x = torch.cat((self.x, self.x2), 0)
            self.y = torch.cat((self.y, torch.tensor(
                [i+1]*len((path/files[i+1]).ls()))))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def train_val_loader(dataset, args):
    if args.mode == "train":
        indices = torch.randperm(len(dataset))
        split = int(np.floor((args.valid_per)*(len(dataset))))
        t_idx, v_idx = indices[:split], indices[split:]
        train_sampler = SubsetRandomSampler(t_idx)
        val_sampler = SubsetRandomSampler(v_idx)
        trainloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True, sampler=train_sampler)
        validloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True, sampler=val_sampler)
        return trainloader, validloader
    elif args.mode == "test":
        testloader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
        return testloader

    else:
        print("Invalid mode")


def get_dataloader(args):
    batch_size = args.batch_size
    valid_per = args.valid_per
    num_workers = args.num_workers
    data = args.dataset
    mode = args.mode
    directory = os. getcwd()
    transform = transforms.Compose([
        transforms.RandomResizedCrop((args.im_size, args.im_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),

    ])

    if data == "CIFAR10":
        args.im_size = 32
        if mode == "train":
            dataset = torchvision.datasets.CIFAR10(
                directory, transform=transform, train=True, download=True)

        elif mode == "test":
            dataset = torchvision.datasets.CIFAR10(
                directory, transform=transform, train=False, download=True)
        else:
            print("Invalid mode")

        return train_val_loader(dataset, args)

    elif data == "MNIST":
        args.im_size = 28
        if mode == "train":
            dataset = torchvision.datasets.MNIST(
                directory, transform=transform, train=True, download=True)
        elif mode == "test":
            dataset = torchvision.datasets.MNIST(
                directory, transform=transform, train=False, download=True)

        else:
            print("invalid mode")
        return train_val_loader(dataset, args)
    elif data == "MNIST_F":
        args.im_size = 28
        if mode == "train":
            dataset = torchvision.datasets.FashionMNIST(
                directory, transform=transform, train=True, download=True)
        elif mode == "test":
            dataset = torchvision.datasets.FashionMNIST(
                directory, transform=transform, train=False, download=True)
        else:
            print("Invalid mode")
        return train_val_loader(dataset, args)
    
    elif data == "PLACES":
        args.im_size = 256
        if mode == "train":
            dataset = torchvision.datasets.Places365(
                directory, transform=transform,small = True, split = "train-challenge",download=True)
        elif mode == "test":
            dataset = torchvision.datasets.CelebA(
                directory, transform=transform, small=False, download=True)
        else:
            print("Invalid mode")
        return train_val_loader(dataset, args)
    
    elif data == "CUSTOM":
        if mode == "train":
            dataset = Custom_Dataset(args.train_path)
        elif mode == "test":
            dataset = Custom_Dataset(args.test_path)

        else:
            print("Invalid mode")
        return train_val_loader(dataset, args)
