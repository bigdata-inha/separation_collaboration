'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from sklearn.metrics import roc_auc_score, average_precision_score
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

import random
import numpy as np

import os
import argparse

from models import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
#parser.add_argument('--save_name', type=str, help='saved model name')
parser.add_argument('--rate', default=1, type=float, help='dataset rate')
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--id', default='cifar10', type=str, help='id model')
parser.add_argument('--ood', default='svhn', type=str, help='ood model')
args = parser.parse_args()

#seed setting
seed = args.seed
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

device = 'cuda:0' #if torch.cuda.is_available() else 'cpu'
best_auroc = 0  # best test accuracy
best_aupr = 0  # best test accuracy
best_fpr = 1  # best test accuracy
best_avg = 0
best_avg_epoch = 0
best_acc = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

if args.id == 'cifar10':
    transform_train = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=256, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

elif args.id == 'svhn':
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

    trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    
oe_dataset = torchvision.datasets.ImageFolder(
    root='/mnt/disk2/workspace/Datasets/tiny-imagenet-200/train', transform=transform_train)

oe_loader = torch.utils.data.DataLoader(oe_dataset, batch_size=512, shuffle=True, num_workers=0)

if args.ood == 'cifar10':
    ood_valid_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
elif args.ood == 'svhn':
    ood_valid_dataset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
elif args.ood == 'cifar100':
    ood_valid_dataset = torchvision.datasets.CIFAR100(root='/mnt/disk2/workspace/Datasets/cifar100', train=False, download=True, transform=transform_test)
elif args.ood == 'places365':
    ood_valid_dataset = torchvision.datasets.ImageFolder(root='/mnt/disk2/workspace/Datasets/place365/test_256', transform=transform_test)

ood_valid_loader = torch.utils.data.DataLoader(ood_valid_dataset, batch_size=256, shuffle=False, num_workers=8)
ood_valid_dataset2 = torchvision.datasets.CIFAR100(root='/mnt/disk2/workspace/Datasets/cifar100', train=False, download=True, transform=transform_test)
ood_valid_loader2 = torch.utils.data.DataLoader(ood_valid_dataset2, batch_size=256, shuffle=False, num_workers=8)
softmax = nn.Softmax(dim=1)


# Model
print('==> Building model..')
if args.id == 'cifar10':
    net = torch.load("./check_model/fedavg_pretrained_model_iid_ResNet34_85_concat_flat.pth")
    
elif args.id == 'svhn':
    net = torch.load("./check_model/fedavg_pretrained_model_iid_ResNet34_85_concat_flat_svhn.pth")
elif args.id == 'cifar100':
    net = torch.load("./check_model/naive_cifar100.pth")

net = net.to(device)

for para in net.parameters():
    para.requires_grad = False

weight_list = ['layer3.0.conv1.weight', 'layer3.0.bn1.weight', 'layer3.0.bn1.bias', 'layer3.0.bn1.running_mean', 'layer3.0.bn1.running_var', 'layer3.0.bn1.num_batches_tracked', 'layer3.0.conv2.weight', 'layer3.0.bn2.weight', 'layer3.0.bn2.bias', 'layer3.0.bn2.running_mean', 'layer3.0.bn2.running_var', 'layer3.0.bn2.num_batches_tracked', 'layer3.0.shortcut.0.weight', 'layer3.0.shortcut.1.weight', 'layer3.0.shortcut.1.bias', 'layer3.0.shortcut.1.running_mean', 'layer3.0.shortcut.1.running_var', 'layer3.0.shortcut.1.num_batches_tracked', 'layer3.1.conv1.weight', 'layer3.1.bn1.weight', 'layer3.1.bn1.bias', 'layer3.1.bn1.running_mean', 'layer3.1.bn1.running_var', 'layer3.1.bn1.num_batches_tracked', 'layer3.1.conv2.weight', 'layer3.1.bn2.weight', 'layer3.1.bn2.bias', 'layer3.1.bn2.running_mean', 'layer3.1.bn2.running_var', 'layer3.1.bn2.num_batches_tracked', 'layer3.2.conv1.weight', 'layer3.2.bn1.weight', 'layer3.2.bn1.bias', 'layer3.2.bn1.running_mean', 'layer3.2.bn1.running_var', 'layer3.2.bn1.num_batches_tracked', 'layer3.2.conv2.weight', 'layer3.2.bn2.weight', 'layer3.2.bn2.bias', 'layer3.2.bn2.running_mean', 'layer3.2.bn2.running_var', 'layer3.2.bn2.num_batches_tracked', 'layer3.3.conv1.weight', 'layer3.3.bn1.weight', 'layer3.3.bn1.bias', 'layer3.3.bn1.running_mean', 'layer3.3.bn1.running_var', 'layer3.3.bn1.num_batches_tracked', 'layer3.3.conv2.weight', 'layer3.3.bn2.weight', 'layer3.3.bn2.bias', 'layer3.3.bn2.running_mean', 'layer3.3.bn2.running_var', 'layer3.3.bn2.num_batches_tracked', 'layer3.4.conv1.weight', 'layer3.4.bn1.weight', 'layer3.4.bn1.bias', 'layer3.4.bn1.running_mean', 'layer3.4.bn1.running_var', 'layer3.4.bn1.num_batches_tracked', 'layer3.4.conv2.weight', 'layer3.4.bn2.weight', 'layer3.4.bn2.bias', 'layer3.4.bn2.running_mean', 'layer3.4.bn2.running_var', 'layer3.4.bn2.num_batches_tracked', 'layer3.5.conv1.weight', 'layer3.5.bn1.weight', 'layer3.5.bn1.bias', 'layer3.5.bn1.running_mean', 'layer3.5.bn1.running_var', 'layer3.5.bn1.num_batches_tracked', 'layer3.5.conv2.weight', 'layer3.5.bn2.weight', 'layer3.5.bn2.bias', 'layer3.5.bn2.running_mean', 'layer3.5.bn2.running_var', 'layer3.5.bn2.num_batches_tracked', 'layer4.0.conv1.weight', 'layer4.0.bn1.weight', 'layer4.0.bn1.bias', 'layer4.0.bn1.running_mean', 'layer4.0.bn1.running_var', 'layer4.0.bn1.num_batches_tracked', 'layer4.0.conv2.weight', 'layer4.0.bn2.weight', 'layer4.0.bn2.bias', 'layer4.0.bn2.running_mean', 'layer4.0.bn2.running_var', 'layer4.0.bn2.num_batches_tracked', 'layer4.0.shortcut.0.weight', 'layer4.0.shortcut.1.weight', 'layer4.0.shortcut.1.bias', 'layer4.0.shortcut.1.running_mean', 'layer4.0.shortcut.1.running_var', 'layer4.0.shortcut.1.num_batches_tracked', 'layer4.1.conv1.weight', 'layer4.1.bn1.weight', 'layer4.1.bn1.bias', 'layer4.1.bn1.running_mean', 'layer4.1.bn1.running_var', 'layer4.1.bn1.num_batches_tracked', 'layer4.1.conv2.weight', 'layer4.1.bn2.weight', 'layer4.1.bn2.bias', 'layer4.1.bn2.running_mean', 'layer4.1.bn2.running_var', 'layer4.1.bn2.num_batches_tracked', 'layer4.2.conv1.weight', 'layer4.2.bn1.weight', 'layer4.2.bn1.bias', 'layer4.2.bn1.running_mean', 'layer4.2.bn1.running_var', 'layer4.2.bn1.num_batches_tracked', 'layer4.2.conv2.weight', 'layer4.2.bn2.weight', 'layer4.2.bn2.bias', 'layer4.2.bn2.running_mean', 'layer4.2.bn2.running_var', 'layer4.2.bn2.num_batches_tracked', 'linear.weight', 'linear.bias', 'detector_linear.weight', 'detector_linear.bias', 'detector_linear2.weight', 'detector_linear2.bias']

for name, param in net.named_parameters():
    if name in weight_list:
        param.requires_grad = True
        
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

def get_auroc(scores_id, scores_ood):
    scores = np.concatenate([scores_id, scores_ood])
    labels = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])
    return roc_auc_score(labels, scores)

def get_aupr(scores_id, scores_ood):
    scores = np.concatenate([scores_id, scores_ood])
    labels = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])
    return average_precision_score(labels, scores)

def get_fpr(scores_id, scores_ood):
    recall_level = 0.95
    scores = np.concatenate([scores_id, scores_ood])
    labels = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])
    return fpr_and_fdr_at_recall(labels, scores, recall_level)

def mixup_criterion(criterion, pred, y_a, y_b, lam, device):
    y_a = y_a.type(torch.LongTensor).to(device)
    y_b = y_b.type(torch.LongTensor).to(device)
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    
    for in_set, out_set in zip(trainloader, oe_loader):

        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]
        data, target = data.to(device), target.to(device)

        feature, x, lam, index = net.forward_mixup(data, len(in_set[0]))

        optimizer.zero_grad()

        loss = mixup_criterion(criterion, x[:len(in_set[0])], target, target[index], lam, device=device)
        loss += 0.2 * -(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()

        loss.requires_grad_(True)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        _, predicted = x[:len(in_set[0])].max(1)
        total += target.size(0)
        correct +=  predicted.eq(target).sum().item()
            
        
        
def auroc_check(epoch):
    global best_auroc
    global best_aupr
    global best_fpr
    global best_avg
    global best_avg_epoch
    global best_acc
    
    total = 0
    correct = 0
    id_scores = np.array([])
    ood_scores = np.array([])
    
    with torch.no_grad():
        net.eval()
        for step, batch in enumerate(testloader):
            # input and target
            batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
            feature, logits = net.auroc_test(batch[0]) 
            output = softmax(logits)
            score = output.max(1).values.detach().cpu().numpy()
            id_scores = np.concatenate((score,id_scores))
            
            total += batch[1].size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(batch[1]).sum().item()
    
    #svhn score
    with torch.no_grad():
        net.eval()
        for step, batch in enumerate(ood_valid_loader):
            # input and target
            batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
            feature, logits = net.auroc_test(batch[0])
            output = softmax(logits)
            score = output.max(1).values.detach().cpu().numpy()
            ood_scores = np.concatenate((score,ood_scores))
    
    svhn_auroc = get_auroc(id_scores,ood_scores)
    svhn_aupr = get_aupr(id_scores,ood_scores)
    svhn_fpr = get_fpr(id_scores,ood_scores)
    
    #cifar100 score
    ood_scores = np.array([])
    with torch.no_grad():
        net.eval()
        for step, batch in enumerate(ood_valid_loader2):
            # input and target
            batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
            feature, logits = net.auroc_test(batch[0])
            output = softmax(logits)
            score = output.max(1).values.detach().cpu().numpy()
            ood_scores = np.concatenate((score,ood_scores))
    
    cifar100_auroc = get_auroc(id_scores,ood_scores)
    cifar100_aupr = get_aupr(id_scores,ood_scores)
    cifar100_fpr = get_fpr(id_scores,ood_scores)
    
    acc = 100.*correct/total
    if epoch == 9:
        if args.ood == 'svhn':
            print("SVHN :: AUROC : ", svhn_auroc, "    Acc : ", acc, "  AUPR : ", svhn_aupr, "  FPR : ", svhn_fpr)
            print("CIFAR100 :: AUROC : ", cifar100_auroc, "    Acc : ", acc, "  AUPR : ", cifar100_aupr, "  FPR : ", cifar100_fpr)
        if args.ood == 'cifar10':
            print("CIFAR10 :: AUROC : ", svhn_auroc, "    Acc : ", acc, "  AUPR : ", svhn_aupr, "  FPR : ", svhn_fpr)
            print("CIFAR100 :: AUROC : ", cifar100_auroc, "    Acc : ", acc, "  AUPR : ", cifar100_aupr, "  FPR : ", cifar100_fpr)
    
for epoch in range(start_epoch, start_epoch+10):
    train(epoch)
    auroc_check(epoch)
    scheduler.step()
    if epoch == 9:
        torch.save(net, "./check_model/paper_final_result/mixup_collaboration_resnet34.pth")