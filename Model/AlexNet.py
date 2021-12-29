import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self,num_classes = 1000) -> None:
        super(AlexNet,self).__init__()
        self.conv1 = nn.Conv2d(3,64,11,4,2)
        self.maxpool = nn.MaxPool2d(3,2)
        self.conv2 = nn.Conv2d(64,192,5,1,2,groups=2)
        self.conv3 = nn.Conv2d(192,384,3,1,1,groups=2)
        self.conv4 = nn.Conv2d(384,256,3,1,1,groups=2)
        self.conv5 = nn.Conv2d(256,256,3,1,1,groups=2)
        self.relu = nn.ReLU(True)
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6,4096),
            nn.ReLU(True),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Linear(4096,num_classes)
        )

    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.maxpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    model = AlexNet()
    x = torch.rand(2,3,224,224)
    y = model(x)
    print(y.size())