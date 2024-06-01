from dependencies import *
from datapreprocessing import *
from vectorizer import vectorizer, adapt_vectorizer
from model import Siamese
from predict import predict

# Initialize the vectorizer
vectorizer_ = vectorizer
# Adapt the vectorizer with the training data
# adapt_vectorizer(Q1_train, Q2_train)

# Load the model architecture
loaded_model = Siamese(vectorizer_)
# loaded_model.summary()

# Load the model weights
loaded_model.load_weights('siamese_model.h5')


# Streamlit app
st.title("LSTM-Based Siamese Model for Duplicate Question Pair Detection")

question1 = st.text_input("Enter the first question:")
question2 = st.text_input("Enter the second question:")
# threshold = st.slider("Threshold:", 0.0, 1.0, 0.7)

if st.button("Predict"):
    result = predict(question1, question2, 0.7, loaded_model, verbose=True)
    st.write("Result:", result)
