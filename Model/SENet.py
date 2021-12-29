import torch
import torch.nn as nn
from Block.SEBlock import SE

class BottleBlock(nn.Module):
    def __init__(self,in_ch,out_ch,stride) -> None:
        super(BottleBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_ch,out_ch,1,1,0,bias=False)
        self.conv2 = nn.Conv2d(out_ch,out_ch,3,stride,1,bias=False)
        self.conv3 = nn.Conv2d(out_ch,out_ch * 4,1,1,0,bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.bn3 = nn.BatchNorm2d(out_ch * 4)
        self.se = SE(out_ch * 4,16)
        self.relu = nn.ReLU(True)
        
        self.sidepath = nn.Sequential(
            nn.Conv2d(in_ch,out_ch * 4,1,stride,0,bias=False),
            nn.BatchNorm2d(out_ch * 4)
        )

    def forward(self,x):
        shortcut = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        coef = self.se(x)
        x = x * coef

        x += self.sidepath(shortcut)
        x = self.relu(x)
        return x

class SENet(nn.Module):

    def __init__(self, num_classes) -> object:
        super(SENet, self).__init__()
        self.channels = 64  # out channels from the first convolutional layer

        self.conv1 = nn.Conv2d(1, self.channels, 7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(self.channels)
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        self.conv2_x = self._make_conv_x(channels=64, blocks=3, strides=1)
        self.conv3_x = self._make_conv_x(channels=128, blocks=4, strides=2)
        self.conv4_x = self._make_conv_x(channels=256, blocks=6, strides=2)
        self.conv5_x = self._make_conv_x(channels=512, blocks=3, strides=2)
        self.pool2 = nn.AvgPool2d(7)
        self.relu = nn.ReLU(True)
        self.softmax = nn.Softmax(-1)
        self.fc = nn.Linear(2048, num_classes)  # for 224 * 224 input size

    def _make_conv_x(self, channels, blocks, strides):
        layers = []
        list_strides = [strides] + [1] * (blocks - 1)  # In conv_x groups, the first strides is 2, the others are ones.
        for i in range(len(list_strides)):
            layers.append(BottleBlock(self.channels, channels, list_strides[i]))
            self.channels = channels * 4
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(self.bn(x))
        x = self.pool1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.softmax(self.fc(x))
        return x

class ResNeXtBlock(nn.Module):
    def __init__(self,in_ch,cardinality,group,stride) -> None:
        super(ResNeXtBlock,self).__init__()
        self.group = cardinality * group
        self.conv1 = nn.Conv2d(in_ch,self.group,1,1,0,bias=False)
        self.conv2 = nn.Conv2d(self.group,self.group,3,stride,1,groups=cardinality,bias=False)
        self.conv3 = nn.Conv2d(self.group,self.group * 2,1,1,0,bias=False)
        self.bn1 = nn.BatchNorm2d(self.group)
        self.bn2 = nn.BatchNorm2d(self.group)
        self.bn3 = nn.BatchNorm2d(self.group * 2)
        self.se = SE(self.group * 2,16)
        self.sidepath = nn.Sequential(
            nn.Conv2d(in_ch,self.group * 2,1,stride,0,bias=False),
            nn.BatchNorm2d(self.group * 2)
        )
        self.relu = nn.ReLU(True)

    def forward(self,x):
        shortcut = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        coef = self.se(x)
        x = x * coef
        x += self.sidepath(shortcut)
        x = self.relu(x)
        return x

class SENeXt(nn.Module):
    def __init__(self,cardinality = 32,group = 4,num_classes = 1000) -> None:
        super(SENeXt,self).__init__()
        self.cardinality = cardinality
        self.channels = 64
        self.conv1 = nn.Conv2d(1,self.channels,7,2,3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(3,2,1)
        self.conv2 = self._make_layers(group * 2,3,1)
        self.conv3 = self._make_layers(group * 4,4,2)
        self.conv4 = self._make_layers(group * 8,6,2)
        self.conv5 = self._make_layers(group * 16,3,2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(-1)
        self.fc = nn.Linear(self.channels,num_classes)

    def _make_layers(self,d,blocks,stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResNeXtBlock(self.channels,self.cardinality,d,s))
            self.channels = self.cardinality*d*2
        return nn.Sequential(*layers)

    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x).view(x.size(0),-1)
        x = self.softmax(self.fc(x))
        return x

if __name__ == '__main__':
    model = SENeXt()
    x = torch.rand(2,224,224)
    y = model(x)
    print(y.size())