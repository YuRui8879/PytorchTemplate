import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size,stride = 1,padding = 0,bias = False,is_trans = True) -> None:
        super(CNNBlock,self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding,bias = bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(True)
        self.is_trans = is_trans

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        if self.is_trans:
            x = self.relu(x)
        return x

class ResBlock(nn.Module):
    def __init__(self,in_ch,out_ch,stride = 1) -> None:
        super(ResBlock,self).__init__()
        self.conv1 = CNNBlock(in_ch,out_ch,kernel_size = 3,stride = 1,padding = 1)
        self.conv2 = CNNBlock(out_ch,out_ch,kernel_size = 3,stride = stride,padding = 1,is_trans=False)
        if stride == 2:
            self.convt = CNNBlock(in_ch,out_ch,kernel_size = 1,stride = 2,is_trans=False)
        else:
            self.convt = CNNBlock(in_ch,out_ch,kernel_size = 1,stride = 1,is_trans=False)
        self.relu = nn.ReLU(True)

    def forward(self,x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.convt:
            shortcut = self.convt(shortcut)
        x += shortcut
        x = self.relu(x)
        return x

class BottleBlock(nn.Module):
    def __init__(self,in_ch,mid_ch,out_ch,stride = 1) -> None:
        super(BottleBlock,self).__init__()
        self.conv1 = CNNBlock(in_ch,mid_ch,kernel_size = 1,stride = 1,padding = 0)
        self.conv2 = CNNBlock(mid_ch,mid_ch,kernel_size = 3,stride = 1,padding = 1)
        self.conv3 = CNNBlock(mid_ch,out_ch,kernel_size = 1,stride = stride,padding = 0,is_trans=False)
        if stride == 2:
            self.convt = CNNBlock(in_ch,out_ch,kernel_size = 1,stride = 2,is_trans=False)
        else:
            self.convt = CNNBlock(in_ch,out_ch,kernel_size = 1,stride = 1,is_trans=False)
        self.relu = nn.ReLU(True)

    def forward(self,x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.convt:
            shortcut = self.convt(shortcut)
        x += shortcut
        x = self.relu(x)
        return x 


class ResNet18(nn.Module):
    def __init__(self) -> None:
        super(ResNet18,self).__init__()
        self.conv1 = nn.Conv2d(1,64,7,2,3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3,2)
        self.conv2 = self._make_layer(64,64,2)
        self.conv3 = self._make_layer(64,128,2)
        self.conv4 = self._make_layer(128,256,2)
        self.conv5 = self._make_layer(256,512,2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512,1000)
        self.relu = nn.ReLU(True)

    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

    def _make_layer(self,in_ch,out_ch,num):
        layer = []
        layer.append(ResBlock(in_ch,out_ch))
        for i in range(num - 2):
            layer.append(ResBlock(out_ch,out_ch))
        layer.append(ResBlock(out_ch,out_ch,2))
        return nn.Sequential(*layer)

class ResNet34(nn.Module):
    def __init__(self) -> None:
        super(ResNet34,self).__init__()
        self.conv1 = nn.Conv2d(1,64,7,2,3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3,2)
        self.conv2 = self._make_layer(64,64,3)
        self.conv3 = self._make_layer(64,128,4)
        self.conv4 = self._make_layer(128,256,6)
        self.conv5 = self._make_layer(256,512,3)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512,1000)
        self.relu = nn.ReLU(True)

    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

    def _make_layer(self,in_ch,out_ch,num):
        layer = []
        layer.append(ResBlock(in_ch,out_ch))
        for i in range(num - 2):
            layer.append(ResBlock(out_ch,out_ch))
        layer.append(ResBlock(out_ch,out_ch,2))
        return nn.Sequential(*layer)

class ResNet50(nn.Module):
    def __init__(self) -> None:
        super(ResNet50,self).__init__()
        self.conv1 = nn.Conv2d(1,64,7,2,3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3,2)
        self.conv2 = self._make_layer(64,64,256,3)
        self.conv3 = self._make_layer(256,128,512,4)
        self.conv4 = self._make_layer(512,256,1024,6)
        self.conv5 = self._make_layer(1024,512,2048,3)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048,1000)
        self.relu = nn.ReLU(True)

    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

    def _make_layer(self,in_ch,mid_ch,out_ch,num):
        layer = []
        layer.append(BottleBlock(in_ch,mid_ch,out_ch))
        for i in range(num - 2):
            layer.append(BottleBlock(out_ch,mid_ch,out_ch))
        layer.append(BottleBlock(out_ch,mid_ch,out_ch,2))
        return nn.Sequential(*layer)

class ResNet101(nn.Module):
    def __init__(self) -> None:
        super(ResNet101,self).__init__()
        self.conv1 = nn.Conv2d(1,64,7,2,3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3,2)
        self.conv2 = self._make_layer(64,64,256,3)
        self.conv3 = self._make_layer(256,128,512,4)
        self.conv4 = self._make_layer(512,256,1024,23)
        self.conv5 = self._make_layer(1024,512,2048,3)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048,1000)
        self.relu = nn.ReLU(True)

    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

    def _make_layer(self,in_ch,mid_ch,out_ch,num):
        layer = []
        layer.append(BottleBlock(in_ch,mid_ch,out_ch))
        for i in range(num - 2):
            layer.append(BottleBlock(out_ch,mid_ch,out_ch))
        layer.append(BottleBlock(out_ch,mid_ch,out_ch,2))
        return nn.Sequential(*layer)

class ResNet152(nn.Module):
    def __init__(self) -> None:
        super(ResNet152,self).__init__()
        self.conv1 = nn.Conv2d(1,64,7,2,3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3,2)
        self.conv2 = self._make_layer(64,64,256,3)
        self.conv3 = self._make_layer(256,128,512,8)
        self.conv4 = self._make_layer(512,256,1024,36)
        self.conv5 = self._make_layer(1024,512,2048,3)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048,1000)
        self.relu = nn.ReLU(True)

    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

    def _make_layer(self,in_ch,mid_ch,out_ch,num):
        layer = []
        layer.append(BottleBlock(in_ch,mid_ch,out_ch))
        for i in range(num - 2):
            layer.append(BottleBlock(out_ch,mid_ch,out_ch))
        layer.append(BottleBlock(out_ch,mid_ch,out_ch,2))
        return nn.Sequential(*layer)


if __name__ == '__main__':
    model = ResNet34()
    x = torch.rand(2,112,112)
    y = model(x)
    print(y.shape)