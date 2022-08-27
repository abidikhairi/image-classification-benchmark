from project.datamodules import CIFAR10DataModule, Fruit360DataModule, MNISTDataModule
from project.modules import ResNet, AlexNet, VGG16, VGG19, Inception


datasets = {
    'mnist': (MNISTDataModule(), 10),
    'cifar10': (CIFAR10DataModule(), 10),
    'fruit360': (Fruit360DataModule(), 24),
}

models = {
    'alexnet': AlexNet,
    'resnet': ResNet,
    'vgg_16': VGG16,
    'vgg_19': VGG19,
    'inception': Inception
}

