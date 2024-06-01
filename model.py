from dependencies import *
from vectorizer import vectorizer
from triplet_loss import TripletLoss
from datapreprocessing import train_generator, val_generator

def Siamese(text_vectorizer, vocab_size=37480, d_feature=128):
    # Define the sequential branch
    branch = tf.keras.models.Sequential(name='sequential')
    branch.add(text_vectorizer)
    branch.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=d_feature, name="embedding"))
    branch.add(tf.keras.layers.LSTM(units=d_feature, return_sequences=True, name="LSTM"))
    branch.add(tf.keras.layers.GlobalAveragePooling1D(name="mean"))
    branch.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x), name="out"))
    
    # Define the inputs
    input1 = tf.keras.layers.Input((1,), dtype=tf.string, name='input_1')
    input2 = tf.keras.layers.Input((1,), dtype=tf.string, name='input_2')
    
    # Apply the branch to both inputs
    branch1 = branch(input1)
    branch2 = branch(input2)
    
    # Concatenate the outputs
    conc = tf.keras.layers.Concatenate(axis=1, name='conc_1_2')([branch1, branch2])
    
    # Build the model
    return tf.keras.models.Model(inputs=[input1, input2], outputs=conc, name="SiameseModel")

# Instantiate and summarize the model
model = Siamese(vectorizer, vocab_size=vectorizer.vocabulary_size())
model.build(input_shape=None)
model.summary()
model.get_layer(name='sequential').summary()

def train_model(Siamese, TripletLoss, text_vectorizer, train_dataset, val_dataset, d_feature=128, lr=0.01, train_steps=5):
    # Instantiate the Siamese model
    model = Siamese(
        text_vectorizer,
        vocab_size=len(text_vectorizer.get_vocabulary()), # Set vocab_size to the size of the vocabulary
        d_feature=d_feature
    )
    
    # Compile the model
    model.compile(
        loss=TripletLoss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr)
    )
    
    # Train the model
    model.fit(
        train_dataset,
        epochs=train_steps,
        validation_data=val_dataset
    )

    return model

# Make the Siamese and train_model functions accessible for import
__all__ = ['Siamese', 'train_model']
