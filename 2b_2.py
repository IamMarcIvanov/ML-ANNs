import numpy as np
import pandas as pd

class AttributeException(Exception):
    def __init__(self, exp_type):
        self.type = exp_type
        
        if self.type == 1:
            self.e1()
            
    def e1(self):
        print('Length of hidden_layers or activations is incorrect')
        print('For k hidden layers we must have k + 2 activation functions')
    
        
class NN:
    def __init__(self, X, Y, learning_rate, hidden_layers=[5], activations=['sigmoid']):
        self.input = X
        self.output = np.zeros(Y.shape)
        self.weights = self.set_weights()
        for n_units in hidden_layers:
            weight = np.random.randn(self.input.shape[1], n_units)
            self.weights.append(weight)
        self.y = Y
        self.hidden_layers = hidden_layers
        self.layers = []
        self.activations = activations
        self.lr = learning_rate
        
        checkVal = self.checker()
        if not checkVal[0]:
            raise AttributeException(checkVal[1])
    
    def checker(self):
        allowed_activations = ['sigmoid', 'ReLU', 'tanh', 'LeakyReLU']
        if not len(self.hidden_layers) + 2 == len(self.activations):
            return False, 1
        if not isinstance(self.lr, float):
            return False, 2
        for n_units in self.hidden_layers:
            if not isinstance(n_units, int):
                return False, 3
        for fn in self.activations:
            if fn not in allowed_activations:
                return False, 4
        return True
    
    def layer(self, X, w, activation_fn='sigmoid'):
        return self.activation(np.dot(X, w), activation_fn=activation_fn)
    
    def activation(self, X, activation_fn='sigmoid'):
        if activation_fn == 'sigmoid':
            return 1 / 1 + np.exp(-X)
        elif activation_fn == 'ReLU':
            return (abs(X) + X) / 2
        elif activation_fn == 'ReLU':
            return np.tanh(X)
        elif activation_fn == 'LeakyReLU':
            y1 = ((X > 0) * X)
            y2 = ((X <= 0) * X * 0.01)
            return y1 + y2
    
    def activation_derivative(self, )
    
    def feedforward(self, X, activation_fn='sigmoid'):
        for acts in self.activations[:-1]:
            X = self.layer(X, activation_fn=activation_fn)
            self.layers.append(X)
        self.output = self.layer(self.layers[-1], self.activations[-1])
    
    def backprop(self):
        for layer in layers:
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(
            self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
