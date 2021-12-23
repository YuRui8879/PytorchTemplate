import numpy as np
import random
import os
from torch.utils.data import Dataset
import torch

class OnlineLoader(Dataset):
    def __init__(self,path,mode = 'train') -> None:
        super(OnlineLoader,self).__init__()
        self.basepath = path
        self.mode = mode
        if not (mode == 'train' or mode == 'valid' or mode == 'test'):
            raise Exception('unsupport mode...')
        self.seed = 0
        self.rate_list = [0,0.7,0.8,1]

        self.train_set = None
        self.valid_set = None
        self.test_set = None
        splited_sample = self._split_dataset(os.listdir(path))
        if len(self.rate_list) == 4:
            self.train_set,self.valid_set,self.test_set = splited_sample[0],splited_sample[1],splited_sample[2]
        elif len(self.rate_list) == 3:
            self.train_set,self.test_set = splited_sample[0],splited_sample[1]
        elif len(self.rate_list) == 2:
            self.test_set = splited_sample[0]
        else:
            raise Exception('unsupport set...')
    
    def _split_dataset(self,sample):
        random.seed(self.seed)
        random.shuffle(sample)
        splited_sample = []
        for i in range(len(self.rate_list) - 1):
            splited_sample.append(sample[self.rate_list[i] * len(sample):self.rate_list[i+1] * len(sample)])
        return splited_sample
        
    def _read_data(self,samp): # This method needs to be implemented
        path = os.path.join(self.basepath,samp)
        fea = None
        label = None
        x = torch.FloatTensor(fea)
        y = torch.FloatTensor(label)
        return x,y
    
    def __getitem__(self,index):
        if self.mode == 'train' and self.train_set:
            feature,label = self._read_data(self.train_set[index])
        elif self.mode == 'valid' and self.valid_set:
            feature,label = self._read_data(self.valid_set[index])
        elif self.mode == 'test' and self.test_set:
            feature,label = self._read_data(self.test_set[index])
        else:
            raise Exception('Please check your dataset...')

        return feature,label

    def __len__(self):
        if self.mode == 'train' and self.train_set:
            length = len(self.train_set)
        elif self.mode == 'valid' and self.valid_set:
            length = len(self.valid_set)
        elif self.mode == 'test' and self.test_set:
            length = len(self.test_set)
        else:
            raise Exception('Please check your dataset...')
        return length