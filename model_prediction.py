import numpy as np
import tensorflow as tf
from intents_preprocessing import preprocess_text, generate_response, intents
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from flask import Flask, request, jsonify

# Load the TensorFlow model
model = tf.keras.models.load_model("model.h5")

def generate_response_with_gpt2(question):
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    input_ids = gpt2_tokenizer.encode(question, return_tensors="pt")
    outputs = gpt2_model.generate(input_ids, do_sample=True, max_length=50)
    return gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_response(question):
    # Vectorize the user's question
    question =generate_response.transform([question])

    # Calculate the cosine similarity of the user's question
    question_similarity = cosine_similarity(question, X)

    # Get the index of the most similar intent
    most_similar_index = question_similarity.argmax()

    # Return the corresponding answer
    answer = intents["intents"][most_similar_index]["answer"]
    if answer == "I'm not sure how to help with that. Let me try to find an answer with GPT-2":
        return generate_response_with_gpt2(question)
    else:
        return answer

def classify_intent(question):
    # Use the TensorFlow model to classify the intent
    question = preprocess_text(question)
    intent_prediction = model.predict(question)
    intent_index = np.argmax(intent_prediction)
    return intent_index
