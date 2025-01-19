# CNN-Based Flower Image Classification with Data Augmentations

This project implements a Convolutional Neural Network (CNN) for classifying flower images into various categories. The model leverages data augmentations to improve its generalization and performance on unseen data.

## Overview

Image classification is a fundamental task in computer vision. In this project, a CNN is trained on a dataset of flower images to classify them into different categories. Data augmentations are applied to increase the diversity of the training set and enhance the robustness of the model.

## Features

- Implements a CNN architecture for image classification.
- Uses data augmentation techniques such as:
  - Random rotations
  - Horizontal and vertical flips
  - Zoom and shift transformations
- Trains the model using the augmented dataset.
- Evaluates model performance on validation and test datasets.

## Project Workflow

1. **Data Loading**:
   - Load the flower dataset, organized by categories in subdirectories.
   - Split the data into training, validation, and test sets.

2. **Data Augmentation**:
   - Apply transformations such as flipping, rotating, and scaling to the training images using `ImageDataGenerator` from TensorFlow/Keras.

3. **Model Architecture**:
   - Build a CNN with layers including:
     - Convolutional layers
     - MaxPooling layers
     - Fully connected (dense) layers
     - Dropout for regularization
   - Use ReLU activation functions and softmax for output classification.

4. **Model Training**:
   - Compile the model with an optimizer (e.g., Adam), categorical cross-entropy loss, and accuracy as the metric.
   - Train the model on the augmented dataset.

5. **Evaluation**:
   - Evaluate the model’s performance on validation and test datasets.
   - Display accuracy and loss plots.

6. **Prediction**:
   - Predict classes for new flower images and visualize results.

## Requirements

The following Python libraries are required to run the project:

- TensorFlow/Keras
- NumPy
- Matplotlib
- Pandas
- scikit-learn

Install the dependencies using:

```bash
pip install tensorflow numpy matplotlib pandas scikit-learn
