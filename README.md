## LSTM-Siamese-Duplicate-Question-Detector

This repository contains an implementation of a Long Short-Term Memory (LSTM)-based Siamese neural network model designed for detecting duplicate question pairs. The project utilizes the [Quora Question Pairs dataset](https://www.kaggle.com/c/quora-question-pairs/data) from Kaggle to train and evaluate the model.

### Overview

The goal of this project is to identify pairs of questions that have the same intent or meaning. This can be particularly useful for platforms where users frequently ask similar questions, such as Q&A websites and customer support portals.

### Features

- **Data Preparation:** 
  - Extracts and processes question pairs from the Quora Question Pairs dataset.
  - Splits the data into training and validation sets.

- **Text Vectorization:**
  - Uses TensorFlow's TextVectorization layer to convert text into integer sequences.

- **Siamese Model Architecture:**
  - **Embedding Layer:** Transforms words into dense vectors.
  - **LSTM Layer:** Captures sequential dependencies in the input text.
  - **Global Average Pooling:** Aggregates the sequence into a fixed-size vector.
  - **L2 Normalization:** Normalizes the output vectors to unit length.
  - The model has two identical branches (hence, Siamese) which process the input question pairs separately but identically. The outputs from both branches are concatenated and used to compute the similarity between the two input questions.

- **Triplet Loss Function:**
  - **Purpose:** The Triplet Loss function is designed to ensure that the distance between the embeddings of duplicate questions (positive pairs) is smaller than the distance between the embeddings of non-duplicate questions (negative pairs) by at least a margin.
  - **Implementation:** The function computes the similarity scores between all pairs in a batch, extracts the positive and negative pairs, and calculates the loss based on the difference between these scores and a specified margin.

- **Training:**
  - Uses TensorFlow's Dataset API to create training and validation datasets.
  - Trains the Siamese model using the Triplet Loss function for improved similarity learning.

- **Evaluation:**
  - Classifies test question pairs based on cosine similarity.
  - Computes accuracy and confusion matrix to assess model performance.

- **Prediction:**
  - Predicts whether two given questions are duplicates using the trained model.

### Getting Started

#### Prerequisites

- Python 3.x
- TensorFlow
- NumPy
- Pandas
- Streamlit

#### Installation

Clone the repository:
```bash
git clone https://github.com/manozpdel/LSTM-Based_Siamese_Model_for_Duplicate_Question_Pair_Detection.git
cd LSTM-Based_Siamese_Model_for_Duplicate_Question_Pair_Detection
```

Install the required packages:
```bash
pip install -r requirements.txt
```

#### Usage

1. **Data Preparation:**
   Download the [Quora Question Pairs dataset](https://www.kaggle.com/c/quora-question-pairs/data) from Kaggle.

2. **Training the Model:**
   To train the model, run:
   ```bash
   python train.py
   ```

3. **Running the Application:**
   To run the application and load the trained model checkpoint, use:
   ```bash
   streamlit run main.py
   ```

### Contributions

Contributions are welcome! Please open an issue or submit a pull request.
