from project.datamodules import CIFAR10DataModule, Fruit360DataModule, MNISTDataModule
from project.modules import DenseNet, ResNet18, AlexNet, VGG11, Inception, VGG13, ResNet34


datasets = {
    'mnist': (MNISTDataModule(), 10),
    'cifar10': (CIFAR10DataModule(), 10),
    'fruit360': (Fruit360DataModule(), 24),
}

models = {
    'alexnet': AlexNet,
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'densenet': DenseNet,
}

