# Pooling_Survey
This is a survey about different pooling methods used in `image classification`

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
  git clone https://github.com/rentainhe/Pooling_Survey.git
  ```
* make data directory for cifar100
  ```bash
  $ cd Pooling_Survey
  $ mkdir data
  ```
 
## Usage

### 1. enter directory
```bash
$ cd Pooling_Survey
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

The supported pooling args are
```
max pooling
average pooling
mixed pooling
```

## Implementated Pooling

- mixed pooling [Mixed pooling for convolutional neural networks](https://rd.springer.com/chapter/10.1007/978-3-319-11740-9_34)
