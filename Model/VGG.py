import torch
from torch.autograd import backward
import torch.nn as nn
from torch.nn.modules.activation import Softmax

class VGGBlock(nn.Module):

    def __init__(self,in_ch,out_ch,kernel_size,stride = 1,padding = 0,bias = False) -> None:
        super(VGGBlock,self).__init__()
        self.conv = nn.Conv2d(in_ch,out_ch,kernel_size = kernel_size,stride = stride,padding = padding,bias = bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(True)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class VGG11(nn.Module):

    def __init__(self) -> None:
        super(VGG11,self).__init__()
        self.maxpool = nn.MaxPool2d(2,2)
        self.conv1 = VGGBlock(1,64,3,1,1)
        self.conv2 = VGGBlock(64,128,3,1,1)
        self.conv3 = VGGBlock(128,256,3,1,1)
        self.conv4 = VGGBlock(256,256,3,1,1)
        self.conv5 = VGGBlock(256,512,3,1,1)
        self.conv6 = VGGBlock(512,512,3,1,1)
        self.conv7 = VGGBlock(512,512,3,1,1)
        self.conv8 = VGGBlock(512,512,3,1,1)
        self.dense = nn.Sequential(
            nn.Linear(25088,4096),
            nn.ReLU(True),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Linear(4096,1000),
            nn.Softmax(-1)
        )

    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.maxpool(x)
        x = x.view(x.size(0),-1)
        x = self.dense(x)
        return x

class VGG13(nn.Module):

    def __init__(self) -> None:
        super(VGG13,self).__init__()
        self.maxpool = nn.MaxPool2d(2,2)
        self.conv1 = VGGBlock(1,64,3,1,1)
        self.conv2 = VGGBlock(64,64,3,1,1)
        self.conv3 = VGGBlock(64,128,3,1,1)
        self.conv4 = VGGBlock(128,128,3,1,1)
        self.conv5 = VGGBlock(128,256,3,1,1)
        self.conv6 = VGGBlock(256,256,3,1,1)
        self.conv7 = VGGBlock(256,512,3,1,1)
        self.conv8 = VGGBlock(512,512,3,1,1)
        self.conv9 = VGGBlock(512,512,3,1,1)
        self.conv10 = VGGBlock(512,512,3,1,1)
        self.dense = nn.Sequential(
            nn.Linear(25088,4096),
            nn.ReLU(True),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Linear(4096,1000),
            nn.Softmax(-1)
        )

    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.maxpool(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.maxpool(x)
        x = x.view(x.size(0),-1)
        x = self.dense(x)
        return x

class VGG16a(nn.Module):

    def __init__(self) -> None:
        super(VGG16a,self).__init__()
        self.maxpool = nn.MaxPool2d(2,2)
        self.conv1 = VGGBlock(1,64,3,1,1)
        self.conv2 = VGGBlock(64,64,3,1,1)
        self.conv3 = VGGBlock(64,128,3,1,1)
        self.conv4 = VGGBlock(128,128,3,1,1)
        self.conv5 = VGGBlock(128,256,3,1,1)
        self.conv6 = VGGBlock(256,256,3,1,1)
        self.conv7 = VGGBlock(256,256,1,1,0)
        self.conv8 = VGGBlock(256,512,3,1,1)
        self.conv9 = VGGBlock(512,512,3,1,1)
        self.conv10 = VGGBlock(512,512,1,1,0)
        self.conv11 = VGGBlock(512,512,3,1,1)
        self.conv12 = VGGBlock(512,512,3,1,1)
        self.conv13 = VGGBlock(512,512,1,1,0)
        self.dense = nn.Sequential(
            nn.Linear(25088,4096),
            nn.ReLU(True),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Linear(4096,1000),
            nn.Softmax(-1)
        )

    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.maxpool(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.maxpool(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.maxpool(x)
        x = x.view(x.size(0),-1)
        x = self.dense(x)
        return x

class VGG16b(nn.Module):

    def __init__(self) -> None:
        super(VGG16b,self).__init__()
        self.maxpool = nn.MaxPool2d(2,2)
        self.conv1 = VGGBlock(1,64,3,1,1)
        self.conv2 = VGGBlock(64,64,3,1,1)
        self.conv3 = VGGBlock(64,128,3,1,1)
        self.conv4 = VGGBlock(128,128,3,1,1)
        self.conv5 = VGGBlock(128,256,3,1,1)
        self.conv6 = VGGBlock(256,256,3,1,1)
        self.conv7 = VGGBlock(256,256,3,1,1)
        self.conv8 = VGGBlock(256,512,3,1,1)
        self.conv9 = VGGBlock(512,512,3,1,1)
        self.conv10 = VGGBlock(512,512,3,1,1)
        self.conv11 = VGGBlock(512,512,3,1,1)
        self.conv12 = VGGBlock(512,512,3,1,1)
        self.conv13 = VGGBlock(512,512,3,1,1)
        self.dense = nn.Sequential(
            nn.Linear(25088,4096),
            nn.ReLU(True),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Linear(4096,1000),
            nn.Softmax(-1)
        )

    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.maxpool(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.maxpool(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.maxpool(x)
        x = x.view(x.size(0),-1)
        x = self.dense(x)
        return x

class VGG19(nn.Module):

    def __init__(self) -> None:
        super(VGG19,self).__init__()
        self.maxpool = nn.MaxPool2d(2,2)
        self.conv1 = VGGBlock(1,64,3,1,1)
        self.conv2 = VGGBlock(64,64,3,1,1)
        self.conv3 = VGGBlock(64,128,3,1,1)
        self.conv4 = VGGBlock(128,128,3,1,1)
        self.conv5 = VGGBlock(128,256,3,1,1)
        self.conv6 = VGGBlock(256,256,3,1,1)
        self.conv7 = VGGBlock(256,256,3,1,1)
        self.conv8 = VGGBlock(256,256,3,1,1)
        self.conv9 = VGGBlock(256,512,3,1,1)
        self.conv10 = VGGBlock(512,512,3,1,1)
        self.conv11 = VGGBlock(512,512,3,1,1)
        self.conv12 = VGGBlock(512,512,3,1,1)
        self.conv13 = VGGBlock(512,512,3,1,1)
        self.conv14 = VGGBlock(512,512,3,1,1)
        self.conv15 = VGGBlock(512,512,3,1,1)
        self.conv16 = VGGBlock(512,512,3,1,1)
        self.dense = nn.Sequential(
            nn.Linear(25088,4096),
            nn.ReLU(True),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Linear(4096,1000),
            nn.Softmax(-1)
        )

    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.maxpool(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxpool(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.maxpool(x)
        x = x.view(x.size(0),-1)
        x = self.dense(x)
        return x

if __name__ == '__main__':
    model = VGG19()
    x = torch.rand(2,224,224)
    y = model(x)
    print(y.size())