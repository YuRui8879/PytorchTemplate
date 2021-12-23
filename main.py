import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.optim.lr_scheduler import MultiStepLR
from Model.VGG import VGG11

from DataLoader.OnlineLoader import OnlineLoader
from Algorithm.EarlyStopping import EarlyStopping
from Metrics.Metrics import Metrics
from Algorithm.Regularization import Regularization

batch_size = 1
learning_rate = 0.001
patience = 10

path = r'.\data'
       
train_loader = Data.DataLoader(OnlineLoader(path,'train'),batch_size=batch_size,shuffle=True,num_workers=0)
valid_loader = Data.DataLoader(OnlineLoader(path,'valid'),batch_size=batch_size,shuffle=False,num_workers=0)
test_loader = Data.DataLoader(OnlineLoader(path,'test'),batch_size=batch_size,shuffle=False,num_workers=0)
        
save_path = r'./'
model = VGG11()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)
early_stopping = EarlyStopping(save_path,patience,verbose = False)
clr = MultiStepLR(optimizer,[20],gamma=0.1)
reg_loss = Regularization(model,0.001)
