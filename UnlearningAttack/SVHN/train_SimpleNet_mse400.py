#! /usr/bin/env python3
import argparse
import json
import os
import pickle

import cv2
import numpy as np
import torch
import random
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import init
# from matplotlib import pyplot as plt
from torchvision import datasets

from model_defination import SimpleNet
from myutlis import select_top_mse_images, load_generated_samples



def load_CIFAR10_from_list(train_path=None, test_path=None, select_path=None):
    trainset = datasets.CIFAR10("./data", train=True, download=True, transform=torchvision.transforms.ToTensor())
    testset = datasets.CIFAR10("./data", train=False, download=True, transform=torchvision.transforms.ToTensor())
    selectset = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor())
    if train_path is not None:
        with open(train_path, 'rb') as f:
            select_list = pickle.load(f)
        trainset.data = torch.index_select(torch.tensor(trainset.data), 0, select_list).numpy()
        trainset.targets = torch.index_select(torch.tensor(trainset.targets), 0, select_list).numpy()

    if test_path is not None:
        with open(test_path, 'rb') as f:
            select_list = pickle.load(f)
        testset.data = torch.index_select(torch.tensor(testset.data), 0, select_list).numpy()
        testset.targets = torch.index_select(torch.tensor(testset.targets), 0, select_list).numpy()

    if select_path is not None:
        with open(select_path, 'rb') as f:
            select_list = pickle.load(f)
        selectset.data = torch.index_select(torch.tensor(selectset.data), 0, select_list).numpy()
        selectset.targets = torch.index_select(torch.tensor(selectset.targets), 0, select_list).numpy()

    return trainset, testset, selectset



def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


def test_in_train(testloader, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            # images, labels = data
            images, labels = data[0].cuda(), data[1].cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            _, pred = torch.max(outputs, 1)
    return correct / total


def save_model(net, PATH):
    torch.save(net.state_dict(), PATH)


def test(testloader, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            # images, labels = data
            images, labels = data[0].cuda(), data[1].cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            _, pred = torch.max(outputs, 1)
    print(correct / total)



def train(trainloader, testloader, net, epoch, resume=False, record=None, save_path=None, RL=False, GA=False):
    net.train()
    if record is not None:
        writer = SummaryWriter(record)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    global_step = 0
    for e in range(epoch):  # loop over the dataset multiple times
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data
            inputs, labels = data[0].cuda(), data[1].cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if RL:
                labels = torch.randint(low=0, high=10, size=labels.size()).cuda()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            if GA:
                loss = 0 - loss
            global_step += 1
            if record is not None:
                writer.add_scalar("loss", loss, global_step=global_step)
            loss.backward()
            optimizer.step()

            if record is not None and i % 40 == 0:
                train_acc = test_in_train(trainloader, net)
                test_acc = test_in_train(testloader, net)
                print(e, train_acc, test_acc)

                writer.add_scalar("train_acc", train_acc, global_step=global_step)
                writer.add_scalar("test_acc", test_acc, global_step=global_step)

            global_step += 1

        if save_path is not None and e % 3 == 0:
            save_model(net, save_path + str(e) + '.pth')

        # print statistics
    if save_path is not None:
        save_model(net, save_path + '.pth')
    print('Finished Training')
    train_acc = test_in_train(trainloader, net)
    test_acc = test_in_train(testloader, net)
    print(train_acc, test_acc)
    return net




if __name__ == "__main__":
    # use the static variables
    model_name = 'SimpleNet'
    model = SimpleNet()
    np.random.seed(0)

    # num_list = random.sample([i for i in range(0,10000)],6000)
    num_list = []
    # print(num_list)

    # imgs_tensor, targets = load_target_samples()

    # train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomCrop(32, 4),
    #         transforms.ToTensor(),
    # ]))
    train_dataset = torchvision.datasets.SVHN(
        root='../data',
        split='train',
        download=True,
        transform=torchvision.transforms.ToTensor()
    )

    test_dataset = torchvision.datasets.SVHN(
        root='../data',
        split='test',
        download=True,
        transform=torchvision.transforms.ToTensor()
    )

    with open('../sample_index_list/SVHN_600.txt', 'rb') as f:
        select_list = pickle.load(f)

    select_list = torch.IntTensor(select_list)
    X_original = torch.index_select(torch.tensor(train_dataset.data), 0, select_list).numpy()
    y = torch.index_select(torch.tensor(train_dataset.labels), 0, select_list).numpy()

    X_generated = load_generated_samples(img_folder="../selected_samples/SVHN/AEwith"+model_name, dataset='SVHN')

    X_generated, y_generated = select_top_mse_images(X_generated, X_original, y, top_n=400)

    train_dataset.data = np.concatenate([train_dataset.data[0:70000], X_original, X_generated], axis=0)
    train_dataset.labels = np.concatenate([train_dataset.labels[0:70000], y, y_generated], axis=0)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


    # 将模型设置为训练模式
    model.cuda()
    # model.load_state_dict(torch.load('./models/ResNet_o&o_GA.pth'))

    model = train(train_loader, test_loader, model, record='./runs/'+  model_name +'_o&g(mse400)',
                  save_path='./models_o&g(mse400)_' ,epoch=300, RL=False, GA=False)
    test(test_loader, model)
