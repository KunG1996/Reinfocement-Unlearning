#! /usr/bin/env python3

import pickle
import numpy as np
import torch
import random
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from torchvision import datasets, models

from Cal_MSE import  top_n_MSE
from model_defination import SimpleNet
from myutlis import load_target_samples, load_generated_samples, select_top_mse_images


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

    optimizer = optim.SGD(net.parameters(), lr=0.001, weight_decay=0.01)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5], gamma=0.2)


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

            if record is not None and i % 50 == 0:
                train_acc = test_in_train(trainloader, net)
                test_acc = test_in_train(testloader, net)
                print(e, train_acc, test_acc)

                writer.add_scalar("train_acc", train_acc, global_step=global_step)
                writer.add_scalar("test_acc", test_acc, global_step=global_step)

            global_step += 1

        # scheduler.step()

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
    model.load_state_dict(torch.load('./models/2000/' + model_name + '_o&g(2000)_GA.pth'))

    mse = 'top'
    num = 400

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    # num_list = random.sample([i for i in range(0,10000)],6000)
    num_list = []

    train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, transform= transforms.Compose([
            transforms.ToTensor(),
    ]))

    test_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, transform= transforms.Compose([
            transforms.ToTensor(),
    ]))


    with open('../sample_index_list/CIFAR10_2000.txt', 'rb') as f:
        select_list = pickle.load(f)

    select_list = torch.IntTensor(select_list)
    X_original = torch.index_select(torch.tensor(train_dataset.data), 0, select_list).numpy()
    y = torch.index_select(torch.tensor(train_dataset.targets), 0, select_list).numpy()

    # image_list = top_n_MSE('../selected_samples/SVHN/AEwithCLS/', '../selected_samples/SVHN/original/', 400 )
    # X_generated = load_generated_samples("../selected_samples/CIFAR102000/AEwith" + model_name, 'CIFAR10')
    # y_generated = y
    # unlearning掉了top400的数据，所以此处把bottom200放进finetuning 集
    # if mse == 'top':
    #     X_generated, y_generated = select_top_mse_images(X_generated, X_original, y, top_n=num)
    # if mse == 'bottom':
    #     X_generated, y_generated = select_top_mse_images(X_generated, X_original, y, bottom_n=num)


    train_dataset.data = np.concatenate([train_dataset.data[0:40000], X_original], axis=0)
    train_dataset.targets = np.concatenate([train_dataset.targets[0: 40000], y], axis=0)
    #

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)



    # 将模型设置为训练模式
    model.cuda()

    model = train(train_loader, test_loader, model, record='./runs/' + model_name + '_o&g(2000)_GA_FT',
                  save_path='./models/2000/' + model_name + '_o&g(2000)_GA_FT_' ,epoch=400, RL=False, GA=False)
    save_model(model, './models/2000/' + model_name + '_o&g(2000)_GA_FT.pth')
    test(train_loader, model)
