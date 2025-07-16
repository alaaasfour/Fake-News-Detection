import streamlit as st
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
import numpy as np

# Load the pre-trained BERT model and tokenizer
model = TFBertForSequenceClassification.from_pretrained("./bert-fake-news-model")
tokenizer = BertTokenizer.from_pretrained("./bert-fake-news-model")

st.title("📰 Fake News Detector")

user_input = st.text_area("Enter a news article or headline:")

if st.button("Detect"):
    if user_input:
        if len(user_input) < 30:
            st.warning("⚠️ Please provide more context — full paragraphs or article excerpts work better than short phrases.")
        else:
            encoding = tokenizer(user_input, return_tensors='tf', truncation=True, padding=True, max_length=128)
            outputs = model(encoding)
            probs = tf.nn.softmax(outputs.logits, axis=-1)
            prediction = tf.argmax(probs, axis=1).numpy()[0]
            confidence = np.max(probs.numpy())

            result = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
            st.markdown(f"## {result}")
            st.write(f"Confidence: {confidence:.2%}")
            probs = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]
            fake_prob = probs[0]
            real_prob = probs[1]

            st.write(f"🧪 Confidence — 🔴 Fake News: {fake_prob:.2%}, 🟢 Real News: {real_prob:.2%}")