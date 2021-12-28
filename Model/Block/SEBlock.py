import torch.nn as nn
import torch

class SE(nn.Module):
    def __init__(self,ch,rate):
        super(SE,self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.compress = nn.Conv2d(ch,ch//rate,1,1,0)
        self.excitation = nn.Conv2d(ch//rate,ch,1,1,0)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.squeeze(x)
        x = self.compress(x)
        x = self.relu(x)
        x = self.excitation(x)
        x = self.sigmoid(x)
        return x