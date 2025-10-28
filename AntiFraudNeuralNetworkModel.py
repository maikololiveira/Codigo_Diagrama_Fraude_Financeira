# AntiFraudNeuralNetworkModel.py

"""
Neural Network for Fraud Classification
This script defines and plots the architecture of a simple neural network using Keras.
It generates a PNG file with a visualization of the network structure.
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model

# Model definition
model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))  # Assuming 20 input variables
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary output: fraud (1) or non-fraud (0)

# Generate network architecture image
plot_model(
    model,
    to_file='AntiFraudNeuralNetworkModel.png',
    show_shapes=True,
    show_layer_names=True
)

print("File 'AntiFraudNeuralNetworkModel.png' generated successfully.")

