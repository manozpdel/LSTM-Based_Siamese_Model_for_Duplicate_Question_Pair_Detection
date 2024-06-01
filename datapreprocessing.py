import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from vectorizer import adapt_vectorizer

# Load and preprocess the data
data = pd.read_csv('train.csv')
data = data.dropna()

# Split the data into training and testing sets
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

# Identify the indices of duplicates in the training set
duplicate_index = train_df['is_duplicate'] == 1
duplicate_index = [i for i, x in enumerate(duplicate_index) if x]

# Use the duplicate indices to index the DataFrame correctly
Q1_train = np.array(train_df.iloc[duplicate_index]['question1'])
Q2_train = np.array(train_df.iloc[duplicate_index]['question2'])

# For the test set, directly extract the columns
Q1_test = np.array(test_df['question1'])
Q2_test = np.array(test_df['question2'])
y_test = np.array(test_df['is_duplicate'])

# Splitting the data into training and validation sets
cut_off = int(len(Q1_train) * 0.8)
train_Q1, train_Q2 = Q1_train[:cut_off], Q2_train[:cut_off]
val_Q1, val_Q2 = Q1_train[cut_off:], Q2_train[cut_off:]

# Adapt the vectorizer with the training data
adapt_vectorizer(Q1_train, Q2_train)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices(((train_Q1, train_Q2), tf.constant([1] * len(train_Q1))))
val_dataset = tf.data.Dataset.from_tensor_slices(((val_Q1, val_Q2), tf.constant([1] * len(val_Q1))))

# Set batch size and shuffle the datasets
batch_size = 256
train_generator = train_dataset.shuffle(len(train_Q1), seed=7, reshuffle_each_iteration=True).batch(batch_size=batch_size)
val_generator = val_dataset.shuffle(len(val_Q1), seed=7, reshuffle_each_iteration=True).batch(batch_size=batch_size)

# Make generators and test sets accessible for import
__all__ = ['train_generator', 'val_generator', 'Q1_test', 'Q2_test', 'y_test','Q1_train','Q2_train']

print('Data preprocessing Tested...')
