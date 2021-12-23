import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

class Tester:

    def __init__(self,model,criterion,test_loader,metrics = None,reg = None,is_parallel = False) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if is_parallel:
            self.model = nn.DataParallel(model).to(self.device)
        else:
            self.model = model
        self.criterion = criterion
        self.test_loader = test_loader
        self.metrics = metrics
        self.reg = reg

    def test(self):
        start_time = time.time()
        loss,res = self._cal_batch()
        end_time = time.time()
        print('T_Time: %.3f - Test_loss: %.5f' %(end_time - start_time,loss),end='')
        if res:
            for keys,values in res.items():
                print(' - ' + keys + ': ' + values,end='')
        print('')
        print('Test finished...')

    def _cal_batch(self):

        test_loss = 0
        test_metrics = None

        self.model.eval()

        for test_idx,data in enumerate(self.test_loader,0):
            inputs,labels = data[0].to(self.device),data[1].to(self.device)
            outputs = self.model(inputs)
                
            loss = self.criterion(outputs,labels)

            with torch.no_grad():
                if self.metrics:
                    self.metrics(outputs.cpu().numpy(),labels.cpu().numpy())

            if self.reg:
                loss += self.reg(self.model)

            test_loss += loss.item()
        test_loss /= (test_idx+1)
        test_metrics = self.metrics.mean()
        return test_loss,test_metrics
        