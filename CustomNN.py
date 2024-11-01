import random
import torch
import os
import pandas as pd
import numpy as np
from collections import OrderedDict
from abc import ABC, abstractmethod
from tqdm import tqdm
from functional import Functional as F

class NeuralNetwork:
    """
        Neural Network from Scratch

        Description:
        -------------
        This project implements a basic neural network from scratch using Python. The aim is to provide a thorough understanding of 
        the internal workings of neural networks, including forward propagation, back propagation, and weight updates. The implementation 
        is designed to be educational, offering insight into the mathematics and logic that drive neural networks, without relying 
        on any high-level machine learning libraries such as TensorFlow or PyTorch.

        Features:
        ---------
        - Initialization: Different methods for initializing weights and biases, such as random and Xavier initialization.
        - Forward Propagation: Implementation of linear and activation functions, including ReLU, Sigmoid, Tanh, and Softmax.
        - Back Propagation: Derivation and computation of gradients for weight and bias updates using chain rule and gradient descent.
        - Loss Functions: Implementation of common loss functions like Mean Squared Error (MSE) and Cross-Entropy Loss.
        - Optimization: Simple gradient descent and advanced methods like Adam and RMSprop.
        - Regularization: Techniques such as L2 regularization and Dropout to prevent overfitting.
        - Evaluation Metrics: Accuracy, Precision, Recall, F1 Score, and more for model evaluation.

        Usage:
        ------
        1. **Model Initialization:** 
           Define the architecture of the neural network, including the number of layers, neurons per layer, activation functions, and initialization methods.
        2. **Training:**
           Train the model on a dataset using the specified loss function, optimizer, and evaluation metrics. 
           The training loop includes forward propagation, loss calculation, back propagation, and parameter updates.
        3. **Evaluation:**
           After training, evaluate the model's performance on a separate test set using the defined evaluation metrics.
        4. **Prediction:**
           Use the trained model to make predictions on new data points.

        Example:
        --------
            # Initialize a simple neural network with 3 layers
            nn = CustomNet(input_size=784, layers=[128, 64, 10])
            # Train the network
            nn.train(X_train, y_train, epochs=100, learning_rate=0.01, batch_size=32)
            # Evaluate the network
            accuracy = nn.evaluate(X_test, y_test)
            # Make predictions
            predictions = nn.predict(new_data)

        Dependencies:
        -------------
        - numpy: For numerical operations and matrix manipulations.
        - tqdm: For progress tracking.
        - pandas: For data manipulation.

        Limitations:
        ------------
        - This implementation is for educational purposes and may not be optimized for large-scale data or production use. 
          It is recommended to use well-established libraries for production-level tasks.

        Notes:
        ------
        This implementation is for educational purposes and may not be optimized for large-scale data or production use. 
        It is recommended to use well-established libraries for production-level tasks.
    """
    def __init__(self,n_layers:int,layer_sizes:list,input_dim:int,random_state:int=1):
        # seed the network with a random number
        np.random.seed(random_state)
        self.n_layers = n_layers
        self.layer_sizes = layer_sizes
        self.input_dim = input_dim
        # initialize weights with random values of the hidden layers
        self.synaptic_weights = OrderedDict(
            [(f"linear{i}.weight", np.random.random((layer_sizes[i-1], layer_sizes[i]))) for i in range(1, n_layers)]
        )
        # initialize weights with random values of the input layer
        self.synaptic_weights["linear0.weight"] = np.random.rand(self.input_dim, layer_sizes[0])
        # move the layer to the beginning 
        self.synaptic_weights.move_to_end("linear0.weight", last=False) 

    def __call__(self, input:np.array)->np.array:
        return self.forward_propagate(input)

    def forward_propagate(self, inputs):
        assert inputs.size!=0, f"input must not be empty, got {inputs.size}"
        assert np.isnan(inputs), f"input must be a numpy array of non-NaN values, got {np.isnan(inputs)}"
        assert inputs.shape == (self.input_dim,), f"input must be of shape ({self.input_dim},), got {inputs.shape}"
        assert np.all(isinstance(inputs,float)), f"input must be a numpy array of floats, got {type(inputs)}"
        # convert all the elements to floats
        inputs = inputs.astype(float)
        # we have to implement the basic matrix multiplication rule here
        # if our input dimension is of m nodes, we expect a input dim of m features; then the input matrix will be 
        # an numpy matrix of shape (n,m) where n is the number of samples/batch size
        # now to consume this payload, we have to do the following operation:
        # inputs@self.synaptic_weights["linear0.weight"]; where self.synaptic_weights["linear0.weight"] is a numpy matrix of shape (m,#nodes in layer 1)
        # this logic makes the forward pass valid by the rule of matrix multiplication for all the subsequent layers
        logits = inputs@self.synaptic_weights["linear0.weight"]
        for layer, weights  in self.synaptic_weights.values():
            if layer!="linear0.weight":
                logits = NeuralNetwork.sigmoid(logits@weights)
        return logits
    
    def backward_propagate(
        self,
        training_set_inputs:np.array,
        training_set_outputs:np.array,
        error_val:float,lr_rate:float=0.01
        ):
        adjustments = np.dot(training_set_inputs.T, error_val * F.sigmoid_derivative(training_set_outputs))
        self.synaptic_weights += adjustments
        
    def train(
        self, 
        training_dataloader:torch.utils.data.DataLoader,
        validation_dataloader:torch.utils.data.DataLoader,
        epochs:int,learning_rate:float,dispatch:bool=False,save_dir:str=None
        ):
        # general assertions
        assert dispatch in [True,False], f"dispatch must be a boolean, got {type(dispatch)}"
        assert isinstance(training_dataloader,torch.utils.data.DataLoader), f"training_dataloader must be a torch.utils.data.DataLoader, got {type(training_dataloader)}"
        assert isinstance(validation_dataloader,torch.utils.data.DataLoader), f"validation_dataloader must be a torch.utils.data.DataLoader, got {type(validation_dataloader)}"
        assert isinstance(epochs,int), f"epochs must be an integer, got {type(epochs)}"
        assert isinstance(learning_rate,float), f"learning_rate must be a float, got {type(learning_rate)}"
        assert isinstance(save_dir,str), f"save_dir must be a string, got {type(save_dir)}"
        assert os.path.exists(save_dir), f"NotExistsError: save_dir does not exist, got {save_dir}"
        
        if dispatch and save_dir is None:
            raise ValueError(f"save_dir must be provided if dispatch is True; got {save_dir}")
        
        training_accuracy, validation_accuracy = list(),list()
        training_loss, validation_loss = list(),list()
        for epoch in range(epochs):
            for _, batch in enumerate(training_dataloader):
                training_set_inputs, training_set_outputs = batch
                outputs = self.forward_propagate(training_set_inputs, training_set_outputs)
                loss = F.mse_loss(outputs, training_set_outputs)
                training_accuracy.append(F.evaluate(training_set_inputs, training_set_outputs)),training_loss.append(loss)
                self.backward_propagate(training_set_inputs, training_set_outputs, loss)
                print(f"Epoch: {epoch+1}: loss: {loss}, Accuracy: {F.evaluate(training_set_inputs, training_set_outputs)}%")
            for _, batch in enumerate(validation_dataloader):
                validation_set_inputs, validation_set_outputs = batch
                outputs = self.forward_propagate(validation_set_inputs, validation_set_outputs)
                loss = F.loss(outputs, validation_set_outputs)
                validation_accuracy.append(F.evaluate(validation_set_inputs, validation_set_outputs)),validation_loss.append(loss)    
        # dispatch the output in a csv file if flag is set to True
        if dispatch:
                pd.DataFrame({
                    'training_accuracy':training_accuracy,
                    'training_loss':training_loss,
                    'validation_accuracy':validation_accuracy,
                    'validation_loss':validation_loss
                }).to_csv(os.path.join(save_dir,"training_output.csv"),index=False)
                
    def predict(self, inputs:np.array)->np.array:
        return F.softmax(self.forward_propagate(inputs))