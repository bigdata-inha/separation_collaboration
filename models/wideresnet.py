import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def mixup_feature(x, alpha=1.0, use_cuda=True, id_len=256):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(id_len).cuda()
    else:
        index = torch.randperm(id_len)

    mixed_x = lam * x[:id_len] + (1 - lam) * x[index, :] #batch안에 data 순서만 바꿔서 원래 순서랑 mixup
    mixed_out = torch.cat([mixed_x, x[id_len:]], dim=0)
    
    return mixed_out, lam, index

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        if self.equalInOut:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        
        #backstage collaboration setting
        self.detector_linear = nn.Linear(32768+16384, 128+256) #concat layer1
        self.detector_linear2 = nn.Linear(128+256, 10) # concat layer2
        
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        feature3 = torch.flatten(out,1)
        out = self.block3(out)
        feature4 = torch.flatten(out,1)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        feature = out.view(-1, self.nChannels)
        return feature, self.fc(feature)
    
    def forward_react(self, x, threshold=1e6):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        feature3 = torch.flatten(out,1)
        out = self.block3(out)
        feature4 = torch.flatten(out,1)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.clip(max=threshold)
        feature = out.view(-1, self.nChannels)
        return self.fc(feature)
    
    def forward_activation(self, x): #react threshold 구할때, activation들 추출할때 사용
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        feature3 = torch.flatten(out,1)
        out = self.block3(out)
        feature4 = torch.flatten(out,1)
        out = self.relu(self.bn1(out))
        activation = F.avg_pool2d(out, 8)
        feature = activation.view(-1, self.nChannels)
        return feature, self.fc(feature)
    
    
    # function to extact the multiple features
    def feature_list(self, x):
        out_list = []
        out = self.conv1(x)
        out_list.append(out)
        print("2")
        out = self.block1(out)
        print("3")
        out_list.append(out)
        print("4")
        out = self.block2(out)
        print("5")
        out_list.append(out)
        print("6")
        out = self.block3(out)
        print("7")
        out_list.append(out)
        print("8")
        out = self.relu(self.bn1(out))
        print("9")
        out = F.avg_pool2d(out, 8)
        print("10")
        out = out.view(-1, self.nChannels)

        y = self.fc(out)

        return y, out_list
    

    # function to extact a specific feature
    def intermediate_forward(self, x, layer_index):
        out = self.conv1(x)
        if layer_index == 1:
            out = self.block1(out)
        elif layer_index == 2:
            out = self.block1(out)
            out = self.block2(out)
        elif layer_index == 3:
            out = self.block1(out)
            out = self.block2(out)
            out = self.block3(out)

        return out

    # function to extact the penultimate features
    def penultimate_forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        penultimate = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        y = self.fc(out)

        return y, penultimate
    
    def get_feature(self, x):
        feature1 = self.conv1(x)
        feature2 = self.block1(feature1)
        feature3 = self.block2(feature2)
        feature4 = self.block3(feature3)
        out = self.relu(self.bn1(feature4))
        out = F.avg_pool2d(out, 8)
        feature5 = out.view(-1, self.nChannels)
        return feature1,feature2, feature3,feature4, feature5
    

    def forward_mixup(self, x, id_len):#our method
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        mixed_feature, lam, index = mixup_feature(out,1, True, id_len)
        out = self.block3(mixed_feature)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        feature = out.view(-1, self.nChannels)
        return feature, self.fc(feature), lam, index
    
    
    def auroc_test(self, x): #auroc test
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        feature = out.view(-1, self.nChannels)
        return feature, self.fc(feature)