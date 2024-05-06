#! /usr/bin/env python3
import os
import pickle

import cv2
import numpy as np
import torch
import random
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
# from matplotlib import pyplot as plt
from torchvision import datasets

from model_defination import SimpleNet, ResNet18
from myutlis import load_generated_samples, select_top_mse_images

def load_data_CIFAR(train_num=None, test_num=None, select_num=None, save_list=True):


    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor())
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor())
    selectset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor())


    if train_num is not None:
        select_list = random.sample([i for i in range(0, 40000)], train_num)
        select_list = torch.IntTensor(select_list)
        trainset.data = torch.index_select(torch.tensor(trainset.data), 0, select_list).numpy()
        trainset.targets = torch.index_select(torch.tensor(trainset.targets), 0, select_list).numpy()
        if save_list:
            with open('./sample_index_list/train_list'+ str(train_num) +'.txt', 'wb') as txt:
                pickle.dump(select_list, txt)
        
    if test_num is not None:
        select_list = random.sample([i for i in range(0, 10000)], test_num)
        select_list = torch.IntTensor(select_list)
        testset.data = torch.index_select(torch.tensor(testset.data), 0, select_list).numpy()
        testset.targets = torch.index_select(torch.tensor(testset.targets), 0, select_list).numpy()
        if save_list:
            with open('./sample_index_list/test_list'+ str(test_num) +'.txt', 'wb') as txt:
                pickle.dump(select_list, txt)

    if select_num is not None:
        select_list = random.sample([i for i in range(40000, 49999)], select_num)
        select_list = torch.IntTensor(select_list)
        selectset.data = torch.index_select(torch.tensor(selectset.data), 0, select_list).numpy()
        selectset.targets = torch.index_select(torch.tensor(selectset.targets), 0, select_list).numpy()
        if save_list:
            with open('./sample_index_list/select_list'+ str(select_num) +'.txt', 'wb') as txt:
                pickle.dump(select_list, txt)

    # trainset.data = np.delete(trainset.data, helpful, axis=0)
    # trainset.targets = np.delete(trainset.targets, helpful, axis=0)



    return trainset, testset, selectset

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



def train(trainloader, testloader, net, epoch, resume=False, record=None, save_path=None, RL=False, GA=False, lr=0.001):
    net.train()
    if record is not None:
        writer = SummaryWriter(record)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=lr)

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
            if record:
                writer.add_scalar("loss", loss, global_step=global_step)
            loss.backward()
            optimizer.step()

            if record and i % 40 == 0:
                train_acc = test_in_train(trainloader, net)
                test_acc = test_in_train(testloader, net)
                print(e, train_acc, test_acc)

                writer.add_scalar("train_acc", train_acc, global_step=global_step)
                writer.add_scalar("test_acc", test_acc, global_step=global_step)

            global_step += 1

        if save_path is not None and e % 1 == 0:
            save_model(net, save_path + str(e) + '.pth')

        train_acc = test_in_train(trainloader, net)
        test_acc = test_in_train(testloader, net)
        print(train_acc, test_acc)

        # print statistics
    if save_path is not None:
        save_model(net, save_path + '.pth')
    print('Finished Training')
    train_acc = test_in_train(trainloader, net)
    test_acc = test_in_train(testloader, net)
    print(train_acc, test_acc)
    return net

def test(testloader, net):
    net = net.eval()
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



def load_target_samples(img_folder="./selected_samples/generated/", target_path="./selected_samples/targets.txt"):
    img_list = []
    for file in os.listdir(img_folder):
        # 读取图像
        img_path = os.path.join(img_folder, file)
        img = cv2.imread(img_path)

        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 添加channel维度

        # 添加到列表
        img_list.append(img)

        # 转换为tensor
    imgs_tensor = torch.from_numpy(np.asarray(img_list))

    with open(target_path, 'rb') as f:
        targets = pickle.load(f)
    targets = torch.from_numpy(np.asarray(targets))

    return imgs_tensor, targets



if __name__ == "__main__":
    # use the static variables
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_name = 'ResNet'
    model = ResNet18()
    model.cuda()
    model = model.load_state_dict(torch.load("./models/ResNet_o(600)_GA_FT_24.pth"))

    num_list = []


    train_dataset = torchvision.datasets.CIFAR10(root='../../data', train=True, transform= transforms.Compose([
            transforms.ToTensor(),
    ]))

    with open('../../sample_index_list/CIFAR10_2000.txt', 'rb') as f:
        select_list = pickle.load(f)

    select_list = torch.IntTensor(select_list)
    X_original = torch.index_select(torch.tensor(train_dataset.data), 0, select_list).numpy()
    y = torch.index_select(torch.tensor(train_dataset.targets), 0, select_list).numpy()

    # 加载A'
    X_generated = load_generated_samples("../../selected_samples/CIFAR102000/AEwith" + model_name, 'CIFAR10')

    TA_set = torchvision.datasets.CIFAR10(root='../../data', train=False,
                                          transform=transforms.Compose([transforms.ToTensor()]))
    TA_loader = torch.utils.data.DataLoader(TA_set, batch_size=48, shuffle=False, num_workers=16)

    MF_set = torchvision.datasets.CIFAR10(root='../../data', train=True,
                                          transform=transforms.Compose([transforms.ToTensor()]))
    MF_set.data = np.concatenate([train_dataset.data, X_original], axis=0)
    MF_set.targets = np.concatenate([train_dataset.targets, y], axis=0)
    MF_loader = torch.utils.data.DataLoader(MF_set, batch_size=48, shuffle=False, num_workers=16)

    MF_A_set = torchvision.datasets.CIFAR10(root='../../data', train=True,
                                            transform=transforms.Compose([transforms.ToTensor()]))
    MF_A_set.data = np.concatenate([X_original], axis=0)
    MF_A_set.targets = np.concatenate([y], axis=0)
    MF_A_loader = torch.utils.data.DataLoader(MF_A_set, batch_size=48, shuffle=False, num_workers=16)

    Ap_set = torchvision.datasets.CIFAR10(root='../../data', train=True,
                                          transform=transforms.Compose([transforms.ToTensor()]))
    Ap = load_generated_samples("../../selected_samples/CIFAR102000/AEwith" + model_name, 'CIFAR10')
    Ap_set.data = np.concatenate([Ap], axis=0)
    Ap_set.targets = np.concatenate([y], axis=0)
    Ap_loader = torch.utils.data.DataLoader(Ap_set, batch_size=48, shuffle=False, num_workers=16)



    train_dataset.data = np.concatenate([X_generated], axis=0)
    train_dataset.targets = np.concatenate([y], axis=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=16)

    test_dataset.data = np.concatenate([X_generated], axis=0)
    test_dataset.targets = np.concatenate([y_generated], axis=0)

    model.load_state_dict(torch.load('../models/2000/' + model_name + '_o&g(2000)_RL_FT.pth'))
    model = train(train_loader, TA_loader, model, epoch=3, save_path='../models/2000/'+ model_name+ '_o&g(2000)_GA_',
                  RL=False, GA=True, lr=0.0001)
    # save_model(model, '../models/2000/'+ model_name + '_o&g(2000)_GA.pth')

    print('TA', test(TA_loader, model))
    print('Acc(A‘)', test(Ap_loader, model))
    print('UE', 1 - test(Ap_loader, model))
    print('MF', test(MF_loader, model))
    print('MF_A', test(MF_A_loader, model))









