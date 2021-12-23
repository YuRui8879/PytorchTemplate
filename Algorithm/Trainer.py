import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

class Trainer:

    def __init__(self,model,criterion,optimizer,train_loader,valid_loader = None,metrics = None,early_stopping = None,clr = None,reg = None,is_parallel = False):
        super(Trainer,self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if is_parallel:
            self.model = nn.DataParallel(model).to(self.device)
        else:
            self.model = model.to(self.device)
        self.criterion = criterion
        self.optimzer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.early_stopping = early_stopping
        self.clr = clr
        self.reg = reg
        self.train_metrics = metrics
        self.valid_metrics = metrics
        self.train_loss = []
        self.valid_loss = []

    def train(self,epochs):
        for epoch in range(1,epochs + 1):
            start_time = time.time()
            train_loss,valid_loss,train_res,valid_res = self._cal_batch()
            end_time = time.time()
            print('[%.3f s] Epoch: %d - Train_loss: %.5f' %(end_time - start_time,epoch,train_loss),end='')
            if train_res:
                for keys,values in train_res.items():
                    print(' - ' + keys + ': ' + values,end='')
            print('')
            if self.valid_loader:
                print('Valid_loss: %.5f' %(valid_loss),end='')
                if valid_res:
                    for keys,values in valid_res.items():
                        print(' - ' + keys + ': ' + values,end='')
                print('')
            if self.clr:
                print('当前学习率: %f' %self.optimizer.state_dict()['param_groups'][0]['lr'])
                self.clr.step()
            if self.early_stopping:
                self.early_stopping(valid_loss, self.model)
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    # 结束模型训练
                    break
            self.train_loss.append(train_loss)
            self.valid_loss.append(valid_loss)
        print('Train finished...')
        return self.model

    def _cal_batch(self):
        
        train_loss = 0
        valid_loss = 0
        train_metrics = None
        valid_metrics = None

        self.model.train()

        for train_idx,data in enumerate(self.train_loader,0):
            inputs,labels = data[0].to(self.device),data[1].to(self.device)
            outputs = self.model(inputs)
                
            loss = self.criterion(outputs,labels)

            with torch.no_grad():
                if self.train_metrics:
                    self.train_metrics(outputs.cpu().numpy(),labels.cpu().numpy())

            if self.reg:
                loss += self.reg(self.model)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
        train_loss /= (train_idx+1)
        train_metrics = self.train_metrics.mean()
        
        if self.valid_loader:
            
            self.model.eval()

            for valid_idx,data in enumerate(self.valid_loader,0):
                inputs,labels = data[0].to(self.device),data[1].to(self.device)
                outputs = self.model(inputs)

                with torch.no_grad():
                    if self.valid_metrics:
                        self.valid_metrics(outputs.cpu().numpy(),labels.cpu().numpy())
                    
                loss = self.criterion(outputs,labels)

                if self.reg:
                    loss += self.reg(self.model)

                valid_loss += loss.item()
            valid_loss /= (valid_idx+1)
            valid_metrics = self.valid_metrics.mean()
                
        return train_loss,valid_loss,train_metrics,valid_metrics

    def plot_train_curve(self):
        plt.plot(self.train_loss,label = 'train')
        if self.valid_loader:
            plt.plot(self.valid_loss,label = 'valid')
        plt.legend()
        plt.title('Train Curve')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.show()

