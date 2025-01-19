# Neural Network for Image Classification: Cat vs Non-Cat

This project implements a simple neural network to classify images as either a cat or a non-cat. The project demonstrates the fundamental steps of binary image classification, including data preprocessing, neural network implementation, and visualization of predictions.

## Features

- Binary classification: Cat (1) vs Non-Cat (0).
- Implementation of a shallow neural network with a single hidden layer.
- Gradient descent optimization to minimize the cost function.
- Supports hyperparameter tuning (e.g., learning rates, number of iterations).
- Visualizations of model performance and predictions.

## Project Workflow

1. **Dataset Loading**: 
   - The dataset consists of labeled images for training and testing.
   - Images are processed and loaded using the `lr_utils` module.
   
2. **Data Preprocessing**:
   - Images are reshaped and flattened to form input vectors.
   - Standardize pixel values to a range between 0 and 1 for better training stability.

3. **Neural Network Implementation**:
   - Define the structure of the network with an input layer, a single hidden layer, and an output layer.
   - Implement the forward propagation step using activation functions:
     - Sigmoid activation for the output layer.
   - Perform backward propagation to compute gradients and update parameters.

4. **Model Training**:
   - Optimize the model using gradient descent.
   - Use the cost function to evaluate training performance.
   - Support for hyperparameter tuning, including learning rate and number of iterations.

5. **Evaluation**:
   - Calculate training and test set accuracy.
   - Analyze misclassified examples to improve model performance.

6. **Visualization**:
   - Plot the cost function to monitor convergence during training.
   - Visualize the model's predictions on test samples.

## Requirements

Install the following Python libraries before running the code:

- numpy
- matplotlib
- h5py
- scipy
- PIL (Python Imaging Library)

You can install the required packages using the following command:

```bash
pip install numpy matplotlib h5py scipy pillow
