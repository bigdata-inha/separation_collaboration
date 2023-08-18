import os
import gc
import pickle
import copy
import os.path
import random
from PIL import Image
from tqdm import tqdm
import numpy as np
from models import *

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, Sampler
import torchvision.datasets
from torch import nn, autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import models.wideresnet as wn

from configs import get_args
from augmentations import get_aug, get_aug_uda, get_aug_fedmatch
from datasets import get_dataset

def get_backbone(backbone):
    backbone = eval(f"{backbone}()")

    return backbone

class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNNCifar(nn.Module):
    def __init__(self):
        super(CNNCifar, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x
    
def get_model(name, backbone):
    if name == 'fedgc' and backbone == 'Mnist':
        model = CNNMnist().to('cuda')
    elif name == 'fedgc' and backbone == 'Cifar':
        model = CNNCifar().to('cuda')
    elif name == 'fedgc' and backbone == 'Svhn':
        model = CNNCifar().to('cuda')
    elif name == 'food' and backbone == 'Cifar':
        model = torchvision.models.resnet34(pretrained=False)
    else:
        raise NotImplementedError
    return model


def linear_rampup(current, rampup_length):
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length

def cosine_rampdown(current, rampdown_length):
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def test_img(net_g, data_loader, args):
    net_g.eval()
    test_loss = 0
    correct = 0
    
    for _ , (data, target) in enumerate(data_loader):
        data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum',ignore_index=-1).item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    
    return accuracy, test_loss

def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch , args):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch
    lr = linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr
    if args.lr_rampdown_epochs:
        lr *= cosine_rampdown(epoch, args.lr_rampdown_epochs)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = w_avg[k].float()
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    all_idxs = [i for i in range(len(dataset))]
    dict_users_unlabeled = {}
        
    for i in range(num_users):
        dict_users_unlabeled[i] = set(np.random.choice(all_idxs, int(num_items) , replace=False))
        all_idxs = list(set(all_idxs) - dict_users_unlabeled[i])
        dict_users_unlabeled[i] = dict_users_unlabeled[i]
    return dict_users_unlabeled

def noniid(dataset, num_users):
    num_shards, num_imgs = 2 * num_users, int(len(dataset)/num_users/2)
    idx_shard = [i for i in range(num_shards)]
    dict_users_unlabeled = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(dataset))
    labels = np.arange(len(dataset))  
    
    for i in range(len(dataset)):
        labels[i] = dataset[i][1]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()] #index value
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users_unlabeled[i] = np.concatenate((dict_users_unlabeled[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    for i in range(num_users):
        dict_users_unlabeled[i] = set(dict_users_unlabeled[i])
        dict_users_unlabeled[i] = dict_users_unlabeled[i]

    return dict_users_unlabeled


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        img, labels = self.dataset[self.idxs[item]]
        return img, labels


def main(device, args):
    
    seed = int(args.seed)
    acc_max = 0
    
    #seed allocation
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dataset_kwargs = {
        'dataset':args.dataset,
        'data_dir': args.data_dir,
        'download':args.download,
        'debug_subset_size':args.batch_size if args.debug else None
    }
    dataloader_kwargs = {
        'batch_size': args.batch_size,
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.num_workers,
    }
    dataloader_unlabeled_kwargs = {
        'batch_size': args.batch_size,
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.num_workers,
    }
    
    #CIFAR10 transform
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif args.dataset == 'svhn':
        transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970])
        ])

        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]),
        ])


    dataset_train =get_dataset( 
        transform=transform_train, #Strong + weak aguemntation
        train=True, 
        **dataset_kwargs
    )

    if args.iid == 'iid':
        dict_users_unlabeled = iid(dataset_train, args.num_users)
    else:
        dict_users_unlabeled = noniid(dataset_train, args.num_users)

    # define model
    model_glob = ResNet34()
    #model_glob = wn.WideResNet(40, 10, widen_factor=4) #depth = 40, width =4
    model_glob.to(device)
    
    user_epoch = {}
    accuracy = []
    result = []

    criterion = nn.CrossEntropyLoss()

    #초기 글로벌 모델 성능 test
    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset( 
            transform=transform_test, 
            train=False,
            **dataset_kwargs),
        shuffle=False,
        **dataloader_kwargs
    )
    model_glob.eval()
    acc, _ = test_img(model_glob, test_loader, args)
    accuracy.append(str(acc))
    del test_loader

    print('Round {:3d}, Acc {:.2f}%'.format(0, acc))
    result.append(acc)
    if acc >= acc_max:
        acc_max = acc
    
    for iter in range(args.num_epochs):

        if iter%1==0:
            test_loader = torch.utils.data.DataLoader(
                dataset=get_dataset( 
                    transform=transform_test, 
                    train=False,
                    **dataset_kwargs),
                shuffle=False,
                **dataloader_kwargs
            )
            model_glob.eval()
            acc, _ = test_img(model_glob, test_loader, args)
            accuracy.append(str(acc))
            del test_loader
            gc.collect()
            torch.cuda.empty_cache()

        w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False) #user를 랜덤으로 선택

        for idx in tqdm(idxs_users, desc="the number of client completed ", mininterval=0.01): # client model update
            if idx in user_epoch.keys():
                user_epoch[idx] += 1 
            else:
                user_epoch[idx] = 1

            model_local = copy.deepcopy(model_glob).to(device) #global model upload

            train_loader_unlabeled = torch.utils.data.DataLoader(
                dataset=DatasetSplit(dataset_train, dict_users_unlabeled[idx]),
                shuffle=True,
                **dataloader_unlabeled_kwargs
            )

            optimizer = torch.optim.SGD(model_local.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
            
            model_local.train()

            for local_iter in range(args.local_ep):
                for i, (img, label) in enumerate(train_loader_unlabeled):

                    img, label = img.to(device), label.to(device)
                    optimizer.zero_grad()
                    outputs = model_local(img)
                    loss = criterion(outputs, label)
                    loss.backward()
                    optimizer.step()

            w_locals.append(copy.deepcopy(model_local.state_dict()))

            del model_local
            del train_loader_unlabeled
            gc.collect()
            torch.cuda.empty_cache()
            
        w_glob = FedAvg(w_locals)
        model_glob.load_state_dict(w_glob)
        
        if iter%1==0:
            print('Round {:3d}, Acc {:.2f}%'.format(iter, acc))
            result.append(acc)
            if acc >= acc_max:
                acc_max = acc
            if iter%5==4:
                print("MODEL SAVE & UPDATE")
                torch.save(model_glob,'./check_model/temp_wide_resnet_cifar/'+str(iter)+'.pth')

if __name__ == "__main__":
    args = get_args()
    main(device=args.device, args=args)
    