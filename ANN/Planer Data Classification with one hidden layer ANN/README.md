# Planar Data Classification with One Hidden Layer

This project implements a neural network with a single hidden layer to classify planar data. The dataset resembles a "flower" pattern, with red and blue points representing two different classes.

## Project Structure

- **Logistic Regression Baseline**: Initial classification using logistic regression to establish a baseline performance.
- **Neural Network Model**: A custom neural network with one hidden layer is trained to improve accuracy.
- **Visualization**: Decision boundaries and results are plotted to analyze model performance.

## Features

1. **Dataset**:
   - A "flower-shaped" dataset generated using `load_planar_dataset()`.

2. **Logistic Regression**:
   - Baseline model using `LogisticRegressionCV` from scikit-learn.
   - Accuracy calculated to compare with the neural network.

3. **Neural Network Architecture**:
   - Input layer size (`n_x`): Equal to the number of features.
   - Hidden layer size (`n_h`): Set to 4 (can be varied).
   - Output layer size (`n_y`): Equal to the number of classes.

4. **Implementation**:
   - Forward propagation with activation functions.
   - Cross-entropy loss for error calculation.
   - Backward propagation for parameter updates.
   - Gradient descent for optimization.

5. **Model Evaluation**:
   - Decision boundaries plotted for various hidden layer sizes.
   - Accuracy calculated and displayed for each hidden layer configuration.

## How to Run

1. Install dependencies:
   ```bash
   pip install numpy matplotlib scikit-learn
