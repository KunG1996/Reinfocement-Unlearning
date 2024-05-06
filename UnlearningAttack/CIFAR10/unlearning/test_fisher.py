#! /usr/bin/env python3
import os
import pickle

import cv2
import numpy as np
import torch
import random
import torchvision
import torchvision.transforms as transforms
# from matplotlib import pyplot as plt
from torchvision import datasets

from CIFAR10.unlearning.fisher import fisher_information_martix
from model_defination import ResNet18, SimpleNet
from myutlis import load_generated_samples, select_top_mse_images


def load_CIFAR10_from_list(train_path=None, test_path=None, select_path=None):
    trainset = datasets.CIFAR10("./data", train=True, download=False, transform=torchvision.transforms.ToTensor())
    testset = datasets.CIFAR10("./data", train=False, download=False, transform=torchvision.transforms.ToTensor())
    selectset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transforms.ToTensor())
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
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


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


def fisher(retain_loader, model):

    device = f"cuda:{int(0)}" if torch.cuda.is_available() else "cpu"
    fisher_approximation = fisher_information_martix(model, retain_loader, device)
    for i, parameter in enumerate(model.parameters()):
        noise = torch.sqrt(0.2 / fisher_approximation[i]).clamp(
            max=1e-3
        ) * torch.empty_like(parameter).normal_(0, 1)
        noise = noise * 10 if parameter.shape[-1] == 10 else noise
        print(torch.max(noise))
        parameter.data = parameter.data + noise
    return model

if __name__ == "__main__":
    # use the static variables
    model = SimpleNet()
    model_name = 'SimpleNet'
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    num_list = []

    train_dataset = torchvision.datasets.CIFAR10(root='../../data', train=True, transform=transforms.ToTensor())

    with open('../../sample_index_list/CIFAR10_2000.txt', 'rb') as f:
        select_list = pickle.load(f)

    select_list = torch.IntTensor(select_list)
    X_original = torch.index_select(torch.tensor(train_dataset.data), 0, select_list).numpy()
    y = torch.index_select(torch.tensor(train_dataset.targets), 0, select_list).numpy()

   
    train_dataset.data = np.concatenate([train_dataset.data, X_original], axis=0)
    train_dataset.targets = np.concatenate([train_dataset.targets, y], axis=0)
    #


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=16)


    model.cuda()
    model.load_state_dict(torch.load('../models/2000/'+ model_name +'_o&g(2000).pth'))
    model = fisher(train_loader, model)
    save_model(model, '../models/2000/'+ model_name+ '_o&g(2000)_FF.pth')
