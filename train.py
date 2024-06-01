# Import necessary modules and functions
from dependencies import *
from vectorizer import vectorizer, adapt_vectorizer
from triplet_loss import TripletLoss
from datapreprocessing import train_generator, val_generator, Q1_train, Q2_train
from model import Siamese, train_model

# Set training parameters
train_steps = 2
batch_size = 256

# Adapt the vectorizer with training data
adapt_vectorizer(Q1_train, Q2_train)

# Train the Siamese model with the specified parameters and data generators
model = train_model(
    Siamese, 
    TripletLoss, 
    vectorizer, 
    train_generator, 
    val_generator, 
    train_steps=train_steps
)

# Save the trained model to a file
model.save('siamese_model.h5')
print("Model saved as 'siamese_model.h5'")
