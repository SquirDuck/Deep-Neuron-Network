# Building a Deep Neural Network: Step by Step

(Class Assignment)
This notebook guides you through building a Deep Neural Network (DNN) from scratch using Python and NumPy, without relying on deep learning frameworks like TensorFlow or PyTorch. The goal of this lab is to help you understand the internal mechanisms of DNNs by implementing the fundamental functions yourself.

## Table of Contents

1.  **Packages**: Import necessary libraries such as `numpy`, `h5py`, `matplotlib`, and helper functions from `dnn_utils`.
2.  **Outline**: An overview of the steps to build the neural network, including parameter initialization, forward propagation, cost calculation, backward propagation, and parameter updates.
3.  **Initialization**:
    * **2-layer Neural Network**: Initialize parameters for a 2-layer network (1 hidden layer).
    * **L-layer Neural Network**: Generalize parameter initialization for an L-layer network.
4.  **Forward Propagation Module**:
    * **Linear Forward**: Compute the linear part $Z = W A + b$.
    * **Linear-Activation Forward**: Combine the linear part with an activation function (Sigmoid or ReLU).
    * **L-Layer Model**: Build the complete forward propagation model for L layers.
5.  **Cost Function**: Compute the cross-entropy cost function to evaluate the model's performance.
6.  **Backward Propagation Module**:
    * **Linear Backward**: Calculate gradients for the linear part.
    * **Linear-Activation Backward**: Combine gradients from the activation function and the linear part.
    * **L-Model Backward**: Perform backward propagation for the entire L-layer network.
    * **Update Parameters**: Update weights and biases using Gradient Descent.



