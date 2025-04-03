# Dense Layer - Class definition
import numpy as np

class Dense:
    def __init__(self, n_inputs, n_neurons):
        #init eights and biases
        self.weights = 0.01*np.random.randn(n_inputs,n_neurons)
        print(self.weights)
        self.biaises = np.zeros((1,n_neurons))

    def forward(self,inputs):
        #calculate outputs
        print(inputs)
        self.output = np.dot(inputs,self.weights) +self.biaises
        
       
