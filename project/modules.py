import pytorch_lightning as pl
from torch import nn
from torch import optim
from torch.nn import functional as F    # for torch.nn.functional.cross_entropy/softmax
from torchvision import models


class Module(pl.LightningModule):
    def __init__(self, num_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001, weight_decay=0.0001)


    def forward(self, x):
        return self.cnn(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        
        self.log('train/loss', loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        logits = F.softmax(y_hat, dim=1)
        
        valid_loss = F.cross_entropy(y_hat, y)
        valid_acc = (logits.argmax(dim=1) == y).float().mean()

        self.log('valid/loss', valid_loss)
        self.log('valid/acc', valid_acc)

        return valid_loss
        
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        logits = F.softmax(y_hat, dim=1)
        
        test_loss = F.cross_entropy(y_hat, y)
        test_acc = (logits.argmax(dim=1) == y).float().mean()

        self.log('test/loss', test_loss)
        self.log('test/acc', test_acc)

        return test_loss


class AlexNet(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.cnn = models.alexnet(num_classes=self.num_classes)
        

class VGG16(Module):
    def __init__(self,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.cnn = models.vgg16_bn(num_classes=self.num_classes)
        

class VGG19(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.cnn = models.vgg19_bn(num_classes=self.num_classes)


class ResNet(Module):
    def __init__(self, num_classes, *args, **kwargs) -> None:
        super().__init__(num_classes, *args, **kwargs)

        self.cnn = models.resnet18(num_classes=self.num_classes)

class Inception(Module):
    def __init__(self, num_classes, *args, **kwargs) -> None:
        super().__init__(num_classes, *args, **kwargs)

        self.cnn = models.inception_v3(num_classes=self.num_classes)