## RESNET 18 Experiments
DATA_DIR=./data python train.py --model resnet --dataset mnist
DATA_DIR=./data python train.py --model resnet --dataset cifar10

## VGG 16 Experiments
DATA_DIR=./data python train.py --model vgg_16 --dataset mnist
DATA_DIR=./data python train.py --model vgg_16 --dataset cifar10
DATA_DIR=./data python train.py --model vgg_16 --dataset fruit360

## VGG 19 Experiments
DATA_DIR=./data python train.py --model vgg_19 --dataset mnist
DATA_DIR=./data python train.py --model vgg_19 --dataset cifar10
DATA_DIR=./data python train.py --model vgg_19 --dataset fruit360

## Inception V3 Experiments
DATA_DIR=./data python train.py --model inception --dataset mnist
DATA_DIR=./data python train.py --model inception --dataset cifar10
DATA_DIR=./data python train.py --model inception --dataset fruit360
