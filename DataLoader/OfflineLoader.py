import numpy as np
import random
import os
from torch.utils.data import Dataset
import torch

class OfflineLoader(Dataset):
    def __init__(self,path,mode = 'train') -> None:
        super(OfflineLoader,self).__init__()
        self.basepath = path
        self.mode = mode
        if not (mode == 'train' or mode == 'valid' or mode == 'test'):
            raise Exception('unsupport mode...')
        self.seed = 0
        self.rate_list = [0,0.7,0.8,1]

        self.train_data = None
        self.valid_data = None
        self.test_data = None
        splited_sample = self._split_dataset(os.listdir(path))
        if len(self.rate_list) == 4:
            train_set,valid_set,test_set = splited_sample[0],splited_sample[1],splited_sample[2]
            self.train_data = self._read_data(train_set)
            self.valid_data = self._read_data(valid_set)
            self.test_data = self._read_data(test_set)
        elif len(self.rate_list) == 3:
            train_set,test_set = splited_sample[0],splited_sample[1]
            self.train_data = self._read_data(train_set)
            self.test_data = self._read_data(test_set)
        elif len(self.rate_list) == 2:
            test_set = splited_sample[0]
            self.test_data = self._read_data(test_set)
        else:
            raise Exception('unsupport set...')
    
    def _split_dataset(self,sample) -> list:
        random.seed(self.seed)
        random.shuffle(sample)
        splited_sample = []
        for i in range(len(self.rate_list) - 1):
            splited_sample.append(sample[self.rate_list[i] * len(sample):self.rate_list[i+1] * len(sample)])
        return splited_sample
        
    def _read_data(self,dataset): # This method needs to be implemented
        fea = []
        label = []
        for samp in os.listdir(dataset):
            path = os.path.join(self.basepath,samp)
            # read data
            fea.append(1)
            label.append(1)
        x = torch.FloatTensor(fea)
        y = torch.FloatTensor(label)
        return (x,y)
    
    def __getitem__(self,index):
        if self.mode == 'train' and self.train_data:
            feature,label = self.train_data[0][index],self.train_data[1][index]
        elif self.mode == 'valid' and self.valid_data:
            feature,label = self.valid_data[0][index],self.valid_data[1][index]
        elif self.mode == 'test' and self.test_data:
            feature,label = self.test_data[0][index],self.test_data[1][index]
        else:
            raise Exception('Please check your dataset...')

        return feature,label

    def __len__(self):
        if self.mode == 'train' and self.train_data:
            length = len(self.train_data[0])
        elif self.mode == 'valid' and self.valid_data:
            length = len(self.valid_data[0])
        elif self.mode == 'test' and self.test_data:
            length = len(self.test_data[0])
        else:
            raise Exception('Please check your dataset...')
        return length