# Import the necessary libraries
import streamlit as st # Streamlit for building the web interface
from transformers import TFBertForSequenceClassification, BertTokenizer # Transformers for the BERT model and tokenizer
import tensorflow as tf # TensorFlow for model execution
import numpy as np # NumPy for numerical operations

# Load the fine-tuned BERT model and tokenizer from the local directory
model = TFBertForSequenceClassification.from_pretrained("./bert-fake-news-model")
tokenizer = BertTokenizer.from_pretrained("./bert-fake-news-model")

# Set up the Streamlit app
st.title("📰 Fake News Detector")

# Create a text area for user input
user_input = st.text_area("Enter a news article or headline:")

# Run prediction when the "Detect" button is clicked
if st.button("Detect"):
    if user_input:
        # Encourage users to input longer and more informative text
        if len(user_input) < 30:
            st.warning("⚠️ Please provide more context — full paragraphs or article excerpts work better than short phrases.")
        else:
            # Tokenize the user input for the BERT model
            encoding = tokenizer(user_input, return_tensors='tf', truncation=True, padding=True, max_length=128)
            # Make predictions using the model
            outputs = model(encoding)
            # Apply softmax to get probabilities for each class (Fake = 0, Real = 1)
            probs = tf.nn.softmax(outputs.logits, axis=-1)
            prediction = tf.argmax(probs, axis=1).numpy()[0]
            confidence = np.max(probs.numpy())
            # Display classification result
            result = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
            st.markdown(f"## {result}")
            st.write(f"Confidence: {confidence:.2%}")
            # Display the probabilities for both classes
            probs = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]
            fake_prob = probs[0]
            real_prob = probs[1]
            # Display detailed probabilities
            st.write(f"🧪 Confidence — 🔴 Fake News: {fake_prob:.2%}, 🟢 Real News: {real_prob:.2%}")