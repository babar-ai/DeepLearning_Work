# Handwritten Digit Detection using TensorFlow and Streamlit

This project demonstrates a simple implementation of a neural network to detect handwritten digits from the MNIST dataset. The project is built using TensorFlow and includes a web-based interface created with Streamlit.

## Project Overview

The main objective of this project is to classify handwritten digits (0-9) using a deep learning model. The model is trained on the MNIST dataset, which contains 60,000 training images and 10,000 testing images of handwritten digits.

## Features

1. **Data Standardization**:
   - The pixel values of the images are scaled between 0 and 1 to improve model convergence.

2. **Neural Network Architecture**:
   - Input Layer: `Flatten()` layer to convert the 28x28 matrix into a 1D array.
   - Hidden Layer 1: Dense layer with 128 neurons and ReLU activation.
   - Output Layer: Dense layer with 10 neurons and softmax activation for multi-class classification.

3. **Model Training**:
   - Loss Function: Sparse categorical cross-entropy.
   - Optimizer: Adam.
   - Training is performed for 10 epochs with 20% of the data used for validation.

4. **Evaluation**:
   - Model accuracy is calculated using the `accuracy_score` function from `sklearn.metrics`.

5. **Visualization**:
   - The project includes visualization of input images and predictions.

6. **Streamlit Integration**:
   - A web app is created using Streamlit to showcase predictions on user-provided handwritten digits.

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/handwritten-digit-detection.git
