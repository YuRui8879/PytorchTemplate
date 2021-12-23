from _typeshed import Self
import numpy as np

class Accuracy:
    def __init__(self) -> None:
        pass

    def __call__(self,pred,real):
        assert len(pred) == len(real)
        pred = np.array(pred)
        real = np.array(real)
        acc = np.sum(np.where((pred - real) == 0,1,0))/len(real)
        return 'acc',acc

    def get_return(self):
        return 'acc',0

    def mean(self,value,count):
        return 'acc',value/count

    def description(self):
        name = 'acc'
        return_type = 'float'
        desc = 'Ratio of the number of correctly classified samples to the total number of samples'
        return name,return_type,desc

class ConfusionMatrix:
    def __init__(self,num_class):
        self.num_class = num_class

    def __call__(self,pred,real):
        assert len(pred) == len(real)
        pred = np.array(pred)
        real = np.array(real)
        C = np.zeros((self.num_class,self.num_class))
        for i in range(len(pred)):
            C[real[i],pred[i]] += 1
        return 'confusion_matrix',C

    def get_return(self):
        return 'confusion_matrix',np.zeros((self.num_class,self.num_class))

    def mean(self,value,count):
        return 'confusion_matrix',value

    def description(self):
        name = 'confusion matrix'
        return_type = 'numpy.array(' + str(self.num_class) + ',' + str(self.num_class) + ')'
        desc = 'Confusion matrix, also known as possibility table or error matrix, each column represents the predicted value and each row represents the actual category'
        return name,return_type,desc

class Precision:
    def __init__(self,num_class):
        self.num_class = num_class

    def __call__(self,pred,real):
        assert len(pred) == len(real)
        pred = np.array(pred)
        real = np.array(real)
        pre = np.zeros(self.num_class)
        for classes in range(self.num_class):
            idx1 = np.where(pred == classes,True,False)
            idx2 = np.where(real == classes,True,False)
            TP = np.sum(idx1&idx2)
            FP = np.sum(idx1&(~idx2))
            pre[classes] = TP/(TP + FP)
        return 'precision',pre
    
    def get_return(self):
        return 'precision',np.zeros(self.num_class)

    def mean(self,value,count):
        return 'precision',value/count

    def description(self):
        name = 'precision'
        return_type = 'numpy.array(' + str(self.num_class) + ')'
        desc = 'Proportion of correctly predicted positive in all predicted positive'
        return name,return_type,desc

class Recall:
    def __init__(self,num_class) -> None:
        self.num_class = num_class

    def __call__(self,pred,real):
        assert len(pred) == len(real)
        pred = np.array(pred)
        real = np.array(real)
        rec = np.zeros(self.num_class)
        for classes in range(self.num_class):
            idx1 = np.where(pred == classes,True,False)
            idx2 = np.where(real == classes,True,False)
            TP = np.sum(idx1&idx2)
            idx3 = np.where(pred != classes,True,False)
            FN = np.sum(idx2&idx3)
            rec[classes] = TP/(TP + FN)
        return 'recall',rec

    def get_return(self):
        return 'recall',np.zeros(self.num_class)
    
    def mean(self,value,count):
        return 'recall',value/count

    def description(self):
        name = 'recall'
        return_type = 'numpy.array(' + str(self.num_class) + ')'
        desc = 'Proportion of correctly predicted positive to all actually positive'
        return name,return_type,desc

class F1:
    def __init__(self,num_class) -> None:
        self.num_class = num_class

    def __call__(self,pred,real):
        assert len(pred) == len(real)
        pred = np.array(pred)
        real = np.array(real)
        F1 = np.zeros(self.num_class)
        for classes in range(self.num_class):
            idx1 = np.where(pred == classes,True,False)
            idx2 = np.where(real == classes,True,False)
            TP = np.sum(idx1&idx2)
            idx3 = np.where(pred != classes,True,False)
            FN = np.sum(idx2&idx3)
            FP = np.sum(idx1&(~idx2))
            F1[classes] = 2*TP/(2*TP + FN + FP)
        return 'F1',F1

    def get_return(self):
        return 'F1',np.zeros(self.num_class)

    def mean(self,value,count):
        return 'F1',value/count

    def description(self):
        name = 'F1'
        return_type = 'numpy.array(' + str(self.num_class) + ')'
        desc = 'F1 score can be regarded as a harmonic average of model accuracy and recall'
        return name,return_type,desc

class F:
    def __init__(self,k,num_class) -> None:
        self.k = k
        self.num_class = num_class

    def __call__(self,pred,real):
        assert len(pred) == len(real)
        pred = np.array(pred)
        real = np.array(real)
        F = np.zeros(self.num_class)
        for classes in range(self.num_class):
            idx1 = np.where(pred == classes,True,False)
            idx2 = np.where(real == classes,True,False)
            TP = np.sum(idx1&idx2)
            idx3 = np.where(pred != classes,True,False)
            FN = np.sum(idx2&idx3)
            FP = np.sum(idx1&(~idx2))
            F[classes] = (1+self.k*self.k)*TP/((1+self.k*self.k)*TP + self.k*self.k*FN + FP)
        return 'F',F

    def get_return(self):
        return 'F',np.zeros(self.num_class)

    def mean(self,value,count):
        return 'F',value/count

    def description(self):
        name = 'F'
        return_type = 'numpy.array(' + str(self.num_class) + ')'
        desc = 'k> 0 measures the relative importance of recall to precision. k> 1. Recall has a greater impact; K < 1 has a greater impact on the accuracy'
        return name,return_type,desc

class Kappa:
    def __init__(self,num_class):
        self.num_class = num_class

    def __call__(self,pred,real):
        assert len(pred) == len(real)
        pred = np.array(pred)
        real = np.array(real)
        p0 = np.sum(np.where((pred - real) == 0,1,0))/len(real)
        pe = 0
        for classes in range(self.num_class):
            a = np.sum(np.where(pred == classes,1,0))
            b = np.sum(np.where(real == classes,1,0))
            pe += a*b
        pe = pe/(len(pred) * len(pred))
        kappa = (p0 - pe)/(1-pe)
        return 'kappa',kappa

    def get_return(self):
        return 'kappa',0

    def mean(self,value,count):
        return 'kappa',value/count

    def description(self):
        name = 'kappa'
        return_type = 'float'
        desc = 'Kappa coefficient is used for consistency test and classification accuracy'
        return name,return_type,desc