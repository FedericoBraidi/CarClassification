from torchinfo import summary
import torch.nn as nn
from torchvision import models
import torch

# Old weights with accuracy 76.130%
class FinetuneResnet18(nn.Module):
    def __init__(self, num_classes):
        super(FinetuneResnet18, self).__init__()

        self.model = models.resnet18(pretrained=True)
        
        self.model.fc = nn.Identity()
        
        for child in self.model.children():
            for params in child.parameters():
                params.requires_grad = False
        
        self.fc1 = nn.Linear(512, 2048)
        self.fc2 = nn.Linear(2048, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)

        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.softmax(self.fc2(x), dim=1)

        return x
    
class FinetuneInceptionV1(nn.Module):
    def __init__(self, num_classes):
        super(FinetuneInceptionV1, self).__init__()

        self.model = models.googlenet(pretrained=True)
        
        self.model.fc = nn.Identity()
        
        for child in self.model.children():
            for params in child.parameters():
                params.requires_grad = False
        
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.model(x)

        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.softmax(self.fc2(x), dim=1)

        return x
    
#mod1=models.resnet18(pretrained=True)
mod=FinetuneInceptionV1(1700)

shape = (1, 3, 224, 224)
summary(mod,shape)
#summary(mod1)
