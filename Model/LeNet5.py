import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self,num_classes = 10) -> None:
        super(LeNet5,self).__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.tanh = nn.Tanh()
        self.maxpool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,num_classes)

    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.tanh(self.conv1(x))
        x = self.maxpool(x)
        x = self.tanh(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0),-1)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    model = LeNet5()
    x = torch.rand(2,32,32)
    y = model(x)
    print(y.size())