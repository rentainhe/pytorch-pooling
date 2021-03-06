# Pooling
This is a collection of different pooling methods used in `image classification` `segmentation` `detection`

## Features
* Multi-GPU support
* Easy and Useful Training log file
* Easy to test different pooling method on classification task

## Requirements
* python3.6
* pytorch1.6.0 + cuda10.1
* tensorboard 2.3.0

## Installation
* clone
  ```
  git clone https://github.com/rentainhe/pytorch-pooling.git
  ```
* make data directory for cifar100
  ```bash
  $ cd pytorch-pooling
  $ mkdir data
  ```
 
## Usage

### 1. enter directory
```bash
$ cd pytorch-pooling
```

### 2. dataset
* Only support cifar100 now (Will support Imagenet Later)
* Using cifar100 dataset from torchvision since it's more convinient

### 3. run tensorboard
Install tensorboard
```bash
$ pip install tensorboard
Run tensorboard
$ tensorboard --logdir runs --port 6006 --host localhost
```

### 4. training
Our base backbone is `vgg16` with `batch_normalization`
```bash
$ python3 train.py --run train --name test --pooling max
```

- ```--run={'train','test','visual'}``` to set the mode to be executed

- ```--name=str``` to set the name of this training

- ```--pooling=str```, e.g, `--pooling='max'` to set the __pooling method__ in `vgg16` to be `max_pool2d`

- ```--gpu=str```, e.g, `--gpu='1'` to set the specified GPU for training

The supported pooling args are
```
max pooling
average pooling
mixed pooling
Lp pooling
lip pooling
soft pooling
```

### 5. Add a new pooling method
You should add a new pooling method `pool.py` in `"/Pooling/pooling_method"` and update the `__init__.py` file

### 6. Addition
- lip pooling: the backbone in original paper is `ResNet`, But I use `VggNet` in this repo, so there might be something wrong with the accuracy

## Results
The result I can get from this repo, I train every model with the same hyperparam and I don't use any tricks in this repo.

|dataset|backbone|pooling|acc|epoch(lr = 0.1)|epoch(lr = 0.02)|epoch(lr = 0.004)|epoch(lr = 0.0008)|total epoch|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
|cifar100|vgg16_bn|max|70.89%|60|60|40|40|200|
|cifar100|vgg16_bn|avg|70.56%|60|60|40|40|200|
|cifar100|vgg16_bn|mixed|71.19%|60|60|40|40|200|
|cifar100|vgg16_bn|Lp(p=2)|70.65%|60|60|40|40|200|
|cifar100|vgg16_bn|Lp(p=3)|70.67%|60|60|40|40|200|
|cifar100|vgg16_bn|lip|71.23%|60|60|40|40|200|
|cifar100|vgg16_bn|softpool|__71.39%__|60|60|40|40|200|

## Implementated Pooling

- mixed pooling [Mixed pooling for convolutional neural networks](https://rd.springer.com/chapter/10.1007/978-3-319-11740-9_34)
- Lp pooling [Convolutional neural networks applied to house numbers digit
classification](https://arxiv.org/abs/1204.3968)
- lip pooling [LIP: Local Importance-based Pooling](https://arxiv.org/abs/1908.04156)
- soft pooling [Refining activation downsampling with SoftPool](https://arxiv.org/abs/2101.00440)