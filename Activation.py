# Activation Class
import numpy as np

# Activation Linear function
class Activation_Linear:
    # Forward Pass
    def forward(self, inputs):
        # Calculate output values from input
        self.output = inputs

# Activation function Sigmoid
class Activation_Sigmoid:
    # Forward Pass
    def forward(self, inputs):
        # Calculate output values from input
        self.output = 1 / (1 + np.exp(-inputs))

# Activation function ReLU
class Activation_ReLU:
    # Forward Pass
    def forward(self, inputs):
        # Calculate output values from input
        self.output = np.maximum(0, inputs)

# Softmax Activation Class Definition
class Activation_SoftMax:
    # Forward pass:
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.exp_values = exp_values
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
