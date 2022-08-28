# ## ALEXNET Experiments
DATA_DIR=./data python train.py --model alexnet --dataset mnist
DATA_DIR=./data python train.py --model alexnet --dataset cifar10
DATA_DIR=./data python train.py --model alexnet --dataset fruit360

## RESNET 18 Experiments
DATA_DIR=./data python train.py --model resnet18 --dataset mnist
DATA_DIR=./data python train.py --model resnet18 --dataset cifar10
DATA_DIR=./data python train.py --model resnet18 --dataset fruit360

## RESNET 34 Experiments
DATA_DIR=./data python train.py --model resnet34 --dataset mnist
DATA_DIR=./data python train.py --model resnet34 --dataset cifar10
DATA_DIR=./data python train.py --model resnet34 --dataset fruit360

## DENSENET Experiments
DATA_DIR=./data python train.py --model densenet --dataset fruit360
DATA_DIR=./data python train.py --model densenet --dataset cifar10
DATA_DIR=./data python train.py --model densenet --dataset fruit360
