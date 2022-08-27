import os
import pytorch_lightning as pl
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import random_split, DataLoader


class DataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.root_dir = os.environ['DATA_DIR']

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=6, shuffle=True)

    
    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, num_workers=6, shuffle=False)
    

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=6, shuffle=False)


class MNISTDataModule(DataModule):
    
    def setup(self, stage = None) -> None:
        
        self.batch_size = 64
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=(0.454, 0.455, 0.456), std=(0.551, 0.552, 0.553))
        ])
        
        trainset = datasets.MNIST(root=self.root_dir, train=True, download=True, transform=transform)
        testset = datasets.MNIST(root=self.root_dir, train=False, download=True, transform=transform)

        valid_size = int(0.1 * len(trainset))

        trainset, validset = random_split(trainset, [len(trainset) - valid_size, valid_size])

        self.trainset = trainset
        self.validset = validset
        self.testset = testset

    
class CIFAR10DataModule(DataModule):
    def setup(self, stage = None) -> None:
        
        self.batch_size = 64

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.454, 0.455, 0.456), std=(0.551, 0.552, 0.553))
        ])
        
        trainset = datasets.CIFAR10(root=self.root_dir, train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root=self.root_dir, train=False, download=True, transform=transform)

        valid_size = int(0.1 * len(trainset))

        trainset, validset = random_split(trainset, [len(trainset) - valid_size, valid_size])

        self.trainset = trainset
        self.validset = validset
        self.testset = testset


class Fruit360DataModule(DataModule):
    def setup(self, stage = None) -> None:
        self.batch_size = 32        
        dataset_path = os.path.join(self.root_dir, 'fruits-360')

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.454, 0.455, 0.456), std=(0.551, 0.552, 0.553))
        ])
        
        trainset = datasets.ImageFolder(root=os.path.join(dataset_path, 'Training'), transform=transform)
        validset = datasets.ImageFolder(root=os.path.join(dataset_path, 'Validation'), transform=transform)
        testset = datasets.ImageFolder(root=os.path.join(dataset_path, 'Test'), transform=transform)

        self.trainset = trainset
        self.validset = validset
        self.testset = testset
