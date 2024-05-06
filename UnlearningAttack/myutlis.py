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


def sample_index_list(train_num=None, test_num=None, select_num=None, save_list=True):
    trainset = torchvision.datasets.SVHN(
        root='./data',
        split='train',
        download=False,
        transform=torchvision.transforms.ToTensor()
    )

    testset = torchvision.datasets.SVHN(
        root='./data',
        split='test',
        download=False,
        transform=torchvision.transforms.ToTensor()
    )
    selectset = torchvision.datasets.SVHN(
        root='./data',
        split='train',
        download=False,
        transform=torchvision.transforms.ToTensor()
    )

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=torchvision.transforms.ToTensor())
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor())
    # selectset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=torchvision.transforms.ToTensor())

    if train_num is not None:
        select_list = random.sample([i for i in range(0, 40000)], train_num)
        select_list = torch.IntTensor(select_list)
        trainset.data = torch.index_select(torch.tensor(trainset.data), 0, select_list).numpy()
        trainset.targets = torch.index_select(torch.tensor(trainset.targets), 0, select_list).numpy()
        if save_list:
            with open('./sample_index_list/train_list' + str(train_num) + '.txt', 'wb') as txt:
                pickle.dump(select_list, txt)

    if test_num is not None:
        select_list = random.sample([i for i in range(0, 10000)], test_num)
        select_list = torch.IntTensor(select_list)
        testset.data = torch.index_select(torch.tensor(testset.data), 0, select_list).numpy()
        testset.targets = torch.index_select(torch.tensor(testset.targets), 0, select_list).numpy()
        if save_list:
            with open('./sample_index_list/test_list' + str(test_num) + '.txt', 'wb') as txt:
                pickle.dump(select_list, txt)

    if select_num is not None:
        select_list = random.sample([int(i) for i in range(70000, 73200)], select_num)
        select_list = torch.IntTensor(select_list)
        # selectset.data = torch.index_select(torch.tensor(selectset.data), 0, select_list).numpy()
        # selectset.targets = torch.index_select(torch.tensor(selectset.targets), 0, select_list).numpy()
        if save_list:
            with open('./sample_index_list/select_list_SVHN_' + str(select_num) + '.txt', 'wb') as txt:
                pickle.dump(select_list, txt)

    # trainset.data = np.delete(trainset.data, helpful, axis=0)
    # trainset.targets = np.delete(trainset.targets, helpful, axis=0)

    return trainset, testset, selectset

def load_target_samples(img_folder="./selected_samples/generated/", target_path="./selected_samples/targets.txt"):
    img_list = []
    for i in range(0, len(os.listdir(img_folder))):
        # 读取图像
        img_path = os.path.join(img_folder, str(i)+'.png')
        img = cv2.imread(img_path)

        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 添加channel维度

        # 添加到列表
        img_list.append(img)

        # 转换为tensor
    imgs_tensor = np.asarray(img_list)

    with open(target_path, 'rb') as f:
        targets = pickle.load(f)
    targets = np.asarray(targets)

    return imgs_tensor, targets


def load_generated_samples(img_folder, dataset='CIFAR10'):
    img_list = []
    for i in range(len(os.listdir(img_folder))):   # 此处必须用index+‘.png’来索引，否则取图像的顺序不对
        # 读取图像
        img_path = os.path.join(img_folder, str(i) + '.png')
        img = cv2.imread(img_path)

        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 添加channel维度

        # 添加到列表
        img_list.append(img)

        # 转换为tensor
    imgs_tensor = np.asarray(img_list)

    if dataset == 'CIFAR10':
        return imgs_tensor
    if dataset == 'SVHN':
        # return imgs_tensor.transpose(0, 3, 1, 2)
        return imgs_tensor.transpose(0, 3, 1, 2)




def select_top_mse_images(A, B, Y, top_n=None, bottom_n=None):
    """
    计算A和B中对应图像的MSE，并从A中选取MSE值最大的top_n张图像以及它们对应的标签。

    参数:
    - A: 一个形状为[600, 32, 32, 3]的ndarray，包含600张图像。
    - B: 一个形状为[600, 32, 32, 3]的ndarray，包含600张图像，与A中的图像存在一一对应关系。
    - Y: 一个长度为600的一维数组，包含A中600张图像的对应标签。
    - top_n: 选取MSE值最大的图像数量，默认为400。

    返回:
    - A_top_n: 从A中选取的MSE值最大的top_n张图像。
    - Y_top_n: 从Y中选取的对应这些图像的标签。
    """

    # 计算A和B中对应图像的MSE
    mse_values = ((A - B) ** 2).mean(axis=(1, 2, 3))

    # 获取MSE值最大的top_n个图像的索引
    if top_n is not None:
        indices = np.argsort(mse_values)[-top_n:]
    if bottom_n is not None:
        indices = np.argsort(mse_values)[:bottom_n]


    # 从A中选取MSE值最大的top_n张图像
    A_n = A[indices]

    # 从Y中选取这些图像对应的标签
    Y_n = Y[indices]

    return A_n, Y_n




if __name__ == '__main__':

    sample_index_list(select_num=2000)