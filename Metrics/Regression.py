import numpy as np

class MSE:
    def __init__(self) -> None:
        pass

    def __call__(self,pred,real):
        assert len(pred) == len(real)
        pred = np.array(pred)
        real = np.array(real)
        mse = np.mean(np.power(pred - real,2))
        return 'mse',mse

    def get_return(self):
        return 'mse',0

    def mean(self,value,count):
        return 'mse',value/count

    def description(self):
        name = 'Mean Square Error'
        return_type = 'float'
        desc = 'The mean square error is the average of the sum of squares of the differences between the data and the true value'
        return name,return_type,desc

class MAE:
    def __init__(self) -> None:
        pass

    def __call__(self,pred,real):
        assert len(pred) == len(real)
        pred = np.array(pred)
        real = np.array(real)
        mae = np.mean(np.abs(pred - real))
        return 'mae',mae

    def get_return(self):
        return 'mae',0

    def mean(self,value,count):
        return 'mae',value/count

    def description(self):
        name = 'Mean Absolute Error'
        return_type = 'float'
        desc = 'The average value of absolute deviation refers to the average value of the absolute deviation of each measured value'
        return name,return_type,desc

class PearsonCorrelationCoefficient:
    def __init__(self) -> None:
        pass

    def __call__(self,pred,real):
        assert len(pred) == len(real)
        pred = np.array(pred)
        real = np.array(real)
        
        n = len(pred)
        sum1 = np.sum(pred)
        sum2 = np.sum(real)
        sum1_pow = np.sum(np.power(pred,2))
        sum2_pow = np.sum(np.power(real,2))
        p_sum = np.sum(pred * real)
        num = p_sum - (sum1*sum2)/n
        den = np.sqrt((sum1_pow-np.power(sum1,2)/n)*(sum2_pow-np.power(sum2,2)/n))
        if den == 0:
            return 0

        return 'corr',num/den

    def get_return(self):
        return 'corr',0

    def mean(self,value,count):
        return 'corr',value/count

    def description(self):
        name = 'Pearson Correlation Coefficient'
        return_type = 'float'
        desc = 'Pearson correlation coefficient is used to measure the linear correlation between two variables X and Y'
        return name,return_type,desc