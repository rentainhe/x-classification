# Mac Classification
Classification on CIFAR10 CIFAR100 and ImageNet using Pytorch

## Features
* Two training ways
  * epoch training: training network using __fixed epoches__ _( warmup_epoch/epoch= 1/200 )_
  * step training: training network using __fixed steps__ _( warmup_steps/total_steps= 1000/80000 )_
* Multi-GPU support
* Easy and Useful Training log file
* Support Different Training Schedule _( multi-step / cosine / linear )_

## Requirements
* python3.6
* pytorch1.6.0 + Cuda10.1
* tensorboard 2.3.0

## Installation
* clone
  ```
  git clone https://github.com/rentainhe/Pytorch_Classification.git
  ```

## Usage

### 1. enter directory
```bash
$ cd Pytorch_Classification
```

### 2. dataset
* Only support cifar10 and cifar100 now (Will support Imagenet Later)
* Using cifar10 and cifar100 dataset from torchvision since it's more convinient

### 3. run tensorboard
Install tensorboard
```bash
$ pip install tensorboard
Run tensorboard
$ tensorboard --logdir runs --port 6006 --host localhost
```

### 4. training
Here is an example which trains a `resnet18` on the `cifar100` dataset using `epoch training loop` 
```bash
$ python3 epoch_run.py --dataset cifar100 --model resnet18 --run train
```

Here is an example which trains a `resnet18` on the `cifar100` dataset using `step training loop`
```bash
$ python3 step_run.py --dataset cifar100 --model resnet18 --run train
```

- ```--run={'train','test','visual'}``` to set the mode to be executed

- ```--model=str```, e.g., to set the model for training

- ```--dataset={'cifar10','cifar100','imagenet'}``` to set the dataset for training

Addition:

- ```--version=str```, e.g, ```--version='test'``` to set a name for this training

- ```--gpu=str```, e.g, ```--gpu='2'``` to train the model on specified GPU device, set ```--gpu='1,2,3'``` to use Multi-GPU Training 

- ```--seed=int```, e.g, ```--seed=1020``` to use a fixed seed to initialize the model. Unset it results in random seeds.

- ```--label_smoothing```, e.g, ```--label_smoothing``` to use label smoothing for training

- ```--gradient_accumulation_steps=int```, e.g, ```--gradient_accumulation_steps=32``` to set the gradient steps to reduce the memory use

Specified Addition For `Epoch` Training:

- ```--warmup_epoch=int```, e.g, ```--warmup_epoch=1``` to set the warmup epoches

- ```--epoch=int```, e.g, ```--epoch=200``` to set the total training epoch nums

Specified Addition For `Step` Training:

- ```--warmup_steps=int```, e.g, ```--warmup_steps=1000``` to set the warmup steps

- ```--num_steps=int```, e.g, ```--num_steps=80000``` to set the total steps

- ```--decay_type=str```, e.g, ```--decay_type='cosine'``` to use the `cosine` learning rate decay schedule

The supported net args are:
```
resnet18
resnet34
resnet50
mobilenet
mobilenetv2
shufflenet
```

## Implementated NetWork

- resnet [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385v1)
- mobilenet [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- shufflenet [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083v2)
