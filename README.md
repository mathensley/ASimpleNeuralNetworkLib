# A Simple Neural Network Library for Python

## Overview

This is a simple library I created as part of my studies in deep learning and machine learning. The main goal was to practice key concepts like model initialization, forward propagation, backward propagation, and optimization techniques. It currently supports three models:

1. **Logistic Regression Model**
2. **Two-Layer Neural Network Model**
3. **N-Layer Neural Network Model**

The library is specifically designed for binary classification problems.

## Limitations and Future Improvements

The library is currently in an experimental version, a few planned improvements:

- **Saving and Loading Model Parameters**: At the moment, model parameters are not saved after training. The next step is to allow saving and loading of trained models for later use.
  
- **Custom Activation and Loss Functions**: Currently, there's only support for fixed activation functions (ReLU, Sigmoid) and a fixed loss function (cross-entropy).
  
- **GPU Support**: The library does not yet support GPU acceleration, but in the future, integration with either PyTorch or TensorFlow to speed up training for larger models and datasets is a possibiliyy.