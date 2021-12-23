import numpy as np

class Metrics:
    def __init__(self,metrics = []) -> None:
        self.metrics = metrics
        self.res = {}
        self.call_count = 0
        for m in self.metrics:
            name,value = m.get_return()
            self.res[name] = value

    def __call__(self,pred,real):
        self.call_count += 1
        for m in self.metrics:
            return_name,result = m(pred,real)
            self.res[return_name] += result
        return self.res

    def mean(self):
        res = {}
        for i in range(len(self.metrics)):
            name,value = self.metrics[i].mean(list(self.res.values())[i],self.call_count)
            res[name] = value
        for m in self.metrics:
            name,value = m.get_return()
            self.res[name] = value
        return res

    def showlist(self):
        print('Name - Return - Description')
        print('---------------------------')
        for m in self.metrics:
            name,return_type,desc = m.get_description()
            print(name,' - ', return_type, ' - ', desc)
            print('---------------------------')