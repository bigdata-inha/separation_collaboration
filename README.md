# Internal Separation & Backstage Collaboration
paper title : Out-of-Distribution Detection via Outlier Exposure in Federated Learning
## Abstract
Among various out-of-distribution (OOD) detection methods in neural networks, outlier exposure (OE) using auxiliary data has shown to achieve practical performance. However, existing OE methods are typically assumed to run in a centralized manner, and thus are not feasible for a standard federated learning (FL) setting where each client has low computing power and cannot collect a variety of auxiliary samples. To address this issue, we propose a practical yet realistic FL scenario where only the central server has a large amount of outlier data and a relatively small amount of in-distribution (ID) data is given to each client. For this scenario, we introduce an effective OE-based OOD detection method, called internal separation & backstage collaboration, which makes the best use of many auxiliary outlier samples without sacrificing the ultimate goal of FL, that is, privacy preservation as well as collaborative training performance. The most challenging part is how to make the same effect in our scenario as in joint centralized training with outliers and ID samples. Our main strategy (internal separation) is to jointly train the feature vectors of an internal layer with outliers in the back layers of the global model, while ensuring privacy preservation. We also suggest an collaborative approach (backstage collaboration) where multiple back layers are trained together to detect OOD samples. Our extensive experiments demonstrate that our method shows remarkable detection performance, compared to baseline approaches in the proposed FL scenario.

## Experiment Command
This repository contains experiments reported in the paper for CIFAR-10, SVHN dataset.
We share the FL converged model we trained for reimplementation.

All these experiments can be run using the following command:
### FedAvg to make converged model for CIFAR-10
```
python fedavg.py --data_dir ../data/cifar --backbone Cifar --dataset cifar10 --batch_size 10 --num_epochs 100 --iid iid --seed 10
```
### FedAvg to make converged model for SVHN
```
python fedavg.py --data_dir ../data/svhn --backbone Svhn --dataset svhn --batch_size 10 --num_epochs 100 --iid iid --seed 10
```
### Internal Separation & Backstage Collaboration & Manifold Mixup (ResNet-34) for CIFAR-10
```
python 2step_train_test_mixup_concat_classifier.py --id cifar10 --ood svhn
```
### Internal Separation & Backstage Collaboration & Manifold Mixup (ResNet-34) for SVHN
```
python 2step_train_test_mixup_concat_classifier.py --id svhn --ood cifar10
```
### Internal Separation & Backstage Collaboration & Manifold Mixup (WideResNet) for CIFAR-10
```
python 2step_train_test_mixup_concat_classifier_wideresnet.py --id cifar10 --ood svhn
```
### Internal Separation & Backstage Collaboration & Manifold Mixup (WideResNet) for SVHN
```
python 2step_train_test_mixup_concat_classifier_wideresnet.py --id svhn --ood cifar10
```


### Requirements
python==3.9.7  
pytorch==1.8.1  
torchvision==0.9.1  
numpy==1.19.2
