import tensorflow as tf
import numpy as np

# Set the random seed
tf.random.set_seed(0)

# Initialize the TextVectorization layer
vectorizer = tf.keras.layers.TextVectorization(
    output_mode='int',
    split='whitespace',
    standardize='strip_punctuation'
)

def adapt_vectorizer(Q1_train, Q2_train):
    """
    Adapt the vectorizer on the given training data.
    
    Args:
        Q1_train (np.array): First set of training questions.
        Q2_train (np.array): Second set of training questions.
        
    Returns:
        None
    """
    vectorizer.adapt(np.concatenate((Q1_train, Q2_train)))

print("Okay done...")

# Make vectorizer accessible for import
__all__ = ['vectorizer', 'adapt_vectorizer']

print("Okay done...")