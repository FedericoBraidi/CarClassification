import torch.nn as nn
import torch
import torchvision.models as models

"""
This code creates the building blocks for the Inception and ResNet architecture and then builds the models themselves as an extension of nn.Module
"""

class ResidualBlock(nn.Module):

    """
    This is a pretty straight forward implementation of a Residual Block.
    """    

    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
class ResNet(nn.Module):

    """
    Generic class for ResNet18 and ResNet34, bigger models modify the amount of blocks
    resnet18 = ResNet(ResidualBlock, [2, 2, 2, 2],num_classes=num_classes).to(device)
    resnet34 = ResNet(ResidualBlock, [3, 4, 6, 3],num_classes=num_classes).to(device) 
    """
    
    def __init__(self, block, layers, num_classes):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:

            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
class ConvBlock (nn.Module):

    """
    This is a pretty straight forward implementation of a Convolutional Block. It includes batchnorm and relu as well.
    """

    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,bias=False,batchnorm=False):
        super(ConvBlock,self).__init__()
        
        self.batchnorm_flag=batchnorm
        self.conv2d = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=bias)
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        
        x=self.conv2d(x)
        if self.batchnorm_flag:
            x=self.batchnorm(x)
        x=self.relu(x)
        
        return x
    
class InceptionBlock (nn.Module):
    
    """
    This implements the Inception Block by building on top of the ConvBlock.
    int_nxn is the intermediate output dimension for the branch that does nxn convolution
    """
    
    def __init__(self , in_channels , out_1x1 , int_3x3 , out_3x3 , int_5x5 , out_5x5 , out_1x1_pooling, batchnorm=False):
        super(InceptionBlock,self).__init__()
        
        self.path1 = ConvBlock(in_channels=in_channels,out_channels=out_1x1,kernel_size=1,stride=1,padding=0,batchnorm=batchnorm)
        self.path2 = nn.Sequential(ConvBlock(in_channels=in_channels,out_channels=int_3x3,kernel_size=1,stride=1,padding=0,batchnorm=batchnorm),
                                   ConvBlock(in_channels=int_3x3,out_channels=out_3x3,kernel_size=3,stride=1,padding=1,batchnorm=batchnorm))    #padding is set to have same output dimension as input
        self.path3 = nn.Sequential(ConvBlock(in_channels=in_channels,out_channels=int_5x5,kernel_size=1,stride=1,padding=0,batchnorm=batchnorm),
                                   ConvBlock(in_channels=int_5x5,out_channels=out_5x5,kernel_size=5,stride=1,padding=2,batchnorm=batchnorm))    #padding is set to have same output dimension as input
        self.path4 = nn.Sequential(nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
                                   ConvBlock(in_channels=in_channels,out_channels=out_1x1_pooling,kernel_size=1,stride=1,padding=0,batchnorm=batchnorm))
        
    def forward(self,x):
        
        p1 = self.path1(x)
        p2 = self.path2(x)
        p3 = self.path3(x)
        p4 = self.path4(x)
        
        tot = torch.cat([p1,p2,p3,p4],dim=1)
        
        return tot
    
class InceptionV1 (nn.Module):

    """
    This creates the original InceptionV1 architecture, using ConvBlock and InceptionBlock as building blocks.
    Unfortunately, this was only tried in the beginning but was never really used, since we used the InceptionModified which follows this.
    """

    def __init__(self , in_channels , num_classes):
        super(InceptionV1,self).__init__()
        
        self.conv1 = ConvBlock(in_channels=in_channels,out_channels=64,kernel_size=7,stride=2,padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.localrespnorm1 = nn.LocalResponseNorm(size=10)

        self.conv2 =  nn.Sequential(ConvBlock(in_channels=64,out_channels=64,kernel_size=1,stride=1,padding=0),
                                    ConvBlock(in_channels=64,out_channels=192,kernel_size=3,stride=1,padding=1))
        self.localrespnorm2 = nn.LocalResponseNorm(size=10)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.inception3a = InceptionBlock(in_channels=192,out_1x1=64,int_3x3=96,out_3x3=128,int_5x5=16,out_5x5=32,out_1x1_pooling=32)
        self.inception3b = InceptionBlock(in_channels=256,out_1x1=128,int_3x3=128,out_3x3=192,int_5x5=32,out_5x5=96,out_1x1_pooling=64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.inception4a = InceptionBlock(in_channels=480, out_1x1=192, int_3x3=96, out_3x3=208, int_5x5=16, out_5x5=48, out_1x1_pooling=64)
        
        self.classifier1 = nn.Sequential(nn.AvgPool2d(kernel_size=5,stride=3,padding=1),
                                         ConvBlock(in_channels=512,out_channels=128,kernel_size=1,stride=1,padding=0),
                                         nn.Flatten(),
                                         nn.Linear(in_features=2048,out_features=1024),
                                         nn.ReLU(),
                                         nn.Dropout(p=0.7),
                                         nn.Linear(in_features=1024,out_features=num_classes))
        
        self.inception4b = InceptionBlock(in_channels=512, out_1x1=160, int_3x3=112, out_3x3=224, int_5x5=24, out_5x5=64, out_1x1_pooling=64)
        self.inception4c = InceptionBlock(in_channels=512, out_1x1=128, int_3x3=128, out_3x3=256, int_5x5=24, out_5x5=64, out_1x1_pooling=64)
        self.inception4d = InceptionBlock(in_channels=512, out_1x1=112, int_3x3=144, out_3x3=288, int_5x5=32, out_5x5=64, out_1x1_pooling=64)
        
        self.classifier2 = nn.Sequential(nn.AvgPool2d(kernel_size=5,stride=3,padding=1),
                                         ConvBlock(in_channels=528,out_channels=128,kernel_size=1,stride=1,padding=0),
                                         nn.Flatten(),
                                         nn.Linear(in_features=2048,out_features=1024),
                                         nn.ReLU(),
                                         nn.Dropout(p=0.7),
                                         nn.Linear(in_features=1024,out_features=num_classes))
        
        self.inception4e = InceptionBlock(in_channels=528, out_1x1=256, int_3x3=160, out_3x3=320, int_5x5=32, out_5x5=128, out_1x1_pooling=128)

        self.maxpool4 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.inception5a = InceptionBlock(in_channels=832, out_1x1=256, int_3x3=160, out_3x3=320, int_5x5=32, out_5x5=128, out_1x1_pooling=128)
        self.inception5b = InceptionBlock(in_channels=832, out_1x1=384, int_3x3=192, out_3x3=384, int_5x5=48, out_5x5=128, out_1x1_pooling=128)

        self.avgpool = nn.AvgPool2d(kernel_size = 7 , stride = 1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear( 1024 , num_classes)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.localrespnorm1(x)

        x = self.conv2(x)
        x = self.localrespnorm2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        
        res1 = self.classifier1(x)
        
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        
        res2 = self.classifier2(x)
        
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        res3 = self.fc1(x)

        return res3, res2, res1 


class InceptionModified (nn.Module):

    """
    This is the modified version of the InceptionV1 where there are no auxiliar classifiers.
    """

    def __init__(self , in_channels , num_classes):
        super(InceptionModified,self).__init__()
        
        self.conv1 = ConvBlock(in_channels=in_channels,out_channels=64,kernel_size=7,stride=2,padding=3,batchnorm=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.conv2 =  nn.Sequential(ConvBlock(in_channels=64,out_channels=64,kernel_size=1,stride=1,padding=0,batchnorm=True),
                                    ConvBlock(in_channels=64,out_channels=192,kernel_size=3,stride=1,padding=1,batchnorm=True))
        self.maxpool2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.inception3a = InceptionBlock(in_channels=192,out_1x1=64,int_3x3=96,out_3x3=128,int_5x5=16,out_5x5=32,out_1x1_pooling=32,batchnorm=True)
        self.inception3b = InceptionBlock(in_channels=256,out_1x1=128,int_3x3=128,out_3x3=192,int_5x5=32,out_5x5=96,out_1x1_pooling=64,batchnorm=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.inception4a = InceptionBlock(in_channels=480, out_1x1=192, int_3x3=96, out_3x3=208, int_5x5=16, out_5x5=48, out_1x1_pooling=64,batchnorm=True)
        self.inception4b = InceptionBlock(in_channels=512, out_1x1=160, int_3x3=112, out_3x3=224, int_5x5=24, out_5x5=64, out_1x1_pooling=64,batchnorm=True)
        self.inception4c = InceptionBlock(in_channels=512, out_1x1=128, int_3x3=128, out_3x3=256, int_5x5=24, out_5x5=64, out_1x1_pooling=64,batchnorm=True)
        self.inception4d = InceptionBlock(in_channels=512, out_1x1=112, int_3x3=144, out_3x3=288, int_5x5=32, out_5x5=64, out_1x1_pooling=64,batchnorm=True)
        self.inception4e = InceptionBlock(in_channels=528, out_1x1=256, int_3x3=160, out_3x3=320, int_5x5=32, out_5x5=128, out_1x1_pooling=128,batchnorm=True)

        self.maxpool4 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.inception5a = InceptionBlock(in_channels=832, out_1x1=256, int_3x3=160, out_3x3=320, int_5x5=32, out_5x5=128, out_1x1_pooling=128,batchnorm=True)
        self.inception5b = InceptionBlock(in_channels=832, out_1x1=384, int_3x3=192, out_3x3=384, int_5x5=48, out_5x5=128, out_1x1_pooling=128,batchnorm=True)

        self.avgpool = nn.AvgPool2d(kernel_size = 7 , stride = 1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear( 1024 , num_classes)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)

        return x
    
class SiameseNetwork(nn.Module):
    
    """
    This is the implementation of the Siamese network.
    """

    def __init__(self, feature_extractor, dummy_input_shape=(1, 3, 224, 224), contra_loss=False):
        super(SiameseNetwork, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.feature_extractor = feature_extractor
        
        """
        This was initially used but later removed to help the feature extractor fine tune its features for verification.
        for child in self.feature_extractor.children():
            for params in child.parameters():
                params.requires_grad = False
        """
        if contra_loss: # Here just in case, contrastive loss was not working, so it's not used
            
            for child in self.feature_extractor.children():
                for params in child.parameters():
                    params.requires_grad = True
        
        # Do a forward pass to get the dimension of the output

        with torch.no_grad():
            dummy_input = torch.zeros(dummy_input_shape)  # Shape: (batch_size, channels, height, width)
            dummy_output = self.feature_extractor(dummy_input.to(device))
            feature_size = dummy_output.view(dummy_output.size(0), -1).size(1)

        # Fully connected layers

        self.fc = nn.Sequential(
            nn.Linear(2*feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        
        self.contra_loss=contra_loss

    def forward(self, input1, input2):
    
        embedding1 = self.feature_extractor(input1)
        embedding2 = self.feature_extractor(input2)
    
        if self.contra_loss:
            return embedding1, embedding2
        else:
            output = torch.cat((embedding1, embedding2), 1)
            output = self.fc(output)
            return output
    
class FinetuneResnet18(nn.Module):

    """
    This is the model which employs the pretrained ResNet18
    """

    def __init__(self, num_classes):
        super(FinetuneResnet18, self).__init__()

        # Get pretrained model from Pytorch

        self.model = models.resnet18(pretrained=True)
        
        # Remove the last layer

        self.model.fc = nn.Identity()
        
        # Freeze the layers

        for child in self.model.children():
            for params in child.parameters():
                params.requires_grad = False
        
        # Add the new layers for classification

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

    """
    This works in the same way as the previous model but with the InceptionV1 pretrained
    """

    def __init__(self, num_classes):
        super(FinetuneInceptionV1, self).__init__()

        self.model = models.googlenet(pretrained=True)
        
        self.model.fc = nn.Identity()
        
        for child in self.model.children():
            for params in child.parameters():
                params.requires_grad = False
        
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.model(x)

        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = nn.functional.softmax(self.fc3(x), dim=1)

        return x
    
class MiniResNet(nn.Module):
    def _init_(self, block, layers, num_classes):
            super(MiniResNet, self)._init_()
            self.inplanes = 32
            self.conv1 = nn.Sequential(
                            nn.Conv2d(3, 32, kernel_size = 7, stride = 2, padding = 3),
                            nn.BatchNorm2d(32),
                            nn.ReLU())
            self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
            self.layer0 = self._make_layer(block, 32, layers[0], stride = 1)
            self.layer1 = self._make_layer(block, 64, layers[1], stride = 2)
            self.layer2 = self._make_layer(block, 128, layers[2], stride = 2)
            self.layer3 = self._make_layer(block, 256, layers[3], stride = 2)
            self.avgpool = nn.AvgPool2d(7, stride=1)
            self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes:

                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                    nn.BatchNorm2d(planes),
                )
            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)

    def forward(self, x):
            x = self.conv1(x)
            x = self.maxpool(x)
            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x