import  numpy as np

class Functional:
    """Functional Class for Neural Networks

    Description:
    -------------
    This class provides static methods for various mathematical operations used in neural networks.
    """
    
    @staticmethod
    def sigmoid(x:np.array)->np.array:
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x:np.array)->np.array:
        return x * (1 - x)
    
    @staticmethod
    def mse_loss(input:np.array,labels:np.array)->float:
        return np.mean((input-labels)**2)
    
    def cross_entropy_loss(labels:np.array,input:np.array,w:np.array)->float:
        return  -np.sum(np.log(Functional.sigmoid(input*labels*w)))

    @staticmethod
    def softmax(x:np.array)->np.array:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    @staticmethod
    def evaluate(input:np.array,labels:np.array)->float:
        return np.mean(np.argmax(input, axis=1) == np.argmax(labels, axis=1))