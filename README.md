# Internal Separation & Backstage Collaboration
paper title : Out-of-Distribution Detection via Outlier Exposure in Federated Learning
## Abstract
Among various out-of-distribution (OOD) detection methods in neural networks, outlier exposure (OE) using auxiliary data has shown to achieve practical performance. However, existing OE methods are typically assumed to run in a centralized manner, and thus are not feasible for a standard federated learning (FL) setting where each client has low computing power and cannot collect a variety of auxiliary samples. To address this issue, we propose a practical yet realistic FL scenario where only the central server has a large amount of outlier data and a relatively small amount of in-distribution (ID) data is given to each client. For this scenario, we introduce an effective OE-based OOD detection method, called internal separation & backstage collaboration, which makes the best use of many auxiliary outlier samples without sacrificing the ultimate goal of FL, that is, privacy preservation as well as collaborative training performance. The most challenging part is how to make the same effect in our scenario as in joint centralized training with outliers and ID samples. Our main strategy (internal separation) is to jointly train the feature vectors of an internal layer with outliers in the back layers of the global model, while ensuring privacy preservation. We also suggest an collaborative approach (backstage collaboration) where multiple back layers are trained together to detect OOD samples. Our extensive experiments demonstrate that our method shows remarkable detection performance, compared to baseline approaches in the proposed FL scenario.

## Experiment Command
This repository currently contains experiments reported in the paper for Split CIFAR-10, Split CIFAR-100, Permuted MNIST, 5-Datasets.
All these experiments can be run using the following command:
### Split CIFAR-10
```
python main.py --dataset='cifar10' --nb_cl_f=2 --nb_cl=2 --ts_epochs=120 --ts_lr=0.1  --lr_factor=0.1 --chunk_size=2000 --ra_lambda=5.0
```

### Split CIFAR-100
```
python main.py --dataset='cifar100' --nb_cl_f=10 --nb_cl=10 --ts_epochs=250 --ts_lr=0.1 --lr_factor=0.1 --chunk_size=25000  --ra_lambda=15.0
```

### Split CIFAR-100 using 5-layer AlexNet
```
python main.py --dataset='cifar100_alexnet' --nb_cl_f=10 --nb_cl=10 --ts_epochs=160 --ts_lr=0.1 --lr_factor=0.1 --chunk_size=25000  --ra_lambda=10.0
```

### Permuted MNIST
```
python main_pmnist.py 
```

### 5-Datasets
```
python main_fivedatasets.py
```

### Requirements
python 3.8.5  
pytorch 1.12.0
