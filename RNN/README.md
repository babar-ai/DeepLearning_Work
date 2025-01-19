# Next Word Predictor using LSTM

This repository contains a project on **Next Word Prediction** using a Long Short-Term Memory (LSTM) model, a type of Recurrent Neural Network (RNN) commonly used for sequential data tasks such as Natural Language Processing (NLP).

## Project Overview

The goal of this project is to build a machine learning model capable of predicting the next word in a sentence given a sequence of words as input. This task is essential in applications such as:

- **Autocomplete Systems**: Helping users type faster by suggesting the next word.
- **Chatbots**: Enhancing conversational abilities by predicting user intents.
- **Language Models**: Forming the backbone for advanced AI models like GPT.

The project leverages LSTM networks, which are particularly effective for sequential data due to their ability to learn long-term dependencies.

---

## Features

- Preprocessing textual data for NLP tasks.
- Tokenization and sequence generation for training the LSTM model.
- Building and training an LSTM-based neural network using Keras.
- Evaluating the model's performance and generating predictions.

---

## Dataset

The project utilizes a text dataset, which must be a clean and large corpus of sentences. Example datasets include:

- Books or articles in plain text.
- Conversational datasets for dialogue systems.

The dataset should:

- Be preprocessed to remove special characters and convert text to lowercase.
- Be tokenized to create sequences of words for training.

---

## Dependencies

Ensure the following libraries are installed in your environment:

- Python 3.7+
- TensorFlow (>=2.0)
- Keras
- NumPy
- Pandas
- Matplotlib

To install the dependencies, use:

```bash
pip install tensorflow keras numpy pandas matplotlib
```

---

## Notebook Structure

### 1. **Data Preprocessing**
   - Load and clean the text dataset.
   - Tokenize the text and create word-to-index mappings.
   - Generate input-output pairs for training the model.

### 2. **Model Architecture**
   - Define an LSTM-based neural network using Keras.
   - Model layers:
     - Embedding layer to convert words to dense vectors.
     - LSTM layer for learning sequence patterns.
     - Dense output layer with a softmax activation function for predicting the next word.

### 3. **Training**
   - Compile the model with loss function and optimizer:
     - Loss: Categorical Cross-Entropy
     - Optimizer: Adam
   - Train the model on the tokenized sequences.

### 4. **Evaluation and Prediction**
   - Evaluate the model's performance on a validation dataset.
   - Use the model to predict the next word for given input sequences.

### 5. **Visualization**
   - Visualize training metrics (e.g., loss over epochs).
   - Show sample predictions to demonstrate the model's capabilities.

---

## Usage

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Next_word_Predictor_using_LSTM
   ```

2. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook Next_word_Predictor_using_LSTM.ipynb
   ```

3. **Follow the steps in the notebook** to preprocess data, train the model, and test predictions.

---

## Results

- The model achieves reasonable accuracy in predicting the next word.
- Example predictions:

  | Input Sequence         | Predicted Next Word |
  |------------------------|---------------------|
  | "The quick brown"      | "fox"              |
  | "Artificial Intelligence" | "is"              |

---

## Future Enhancements

- Fine-tune the model using a larger and domain-specific dataset.
- Experiment with bidirectional LSTMs or Transformer-based architectures for improved accuracy.
- Deploy the model as an API using FastAPI or Flask for real-world applications.

---

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request for improvements.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For questions or collaboration, reach out to:

- **Babar Raheem**
- [GitHub Profile](https://github.com/yourusername)
- [LinkedIn](https://linkedin.com/in/yourusername)

---

Happy coding! 