import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sklearn.metrics.pairwise import cosine_similarity

# Load the intents from the json file
with open("data/json/main.json", "r") as file:
    intents = json.load(file)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return " ".join(text)

# Define the function to generate responses
def generate_response(question):
    # Initialize the vectorizer
    vectorizer = CountVectorizer(tokenizer=preprocess_text)

    # Fit the vectorizer to the intents
    X = vectorizer.fit_transform([i["text"] for i in intents["intents"]])
    # Calculate the cosine similarities
    similarity = cosine_similarity(X)

    # Preprocess the user's question
    question = preprocess_text(question)

    # Vectorize the user's question
    question = vectorizer.transform([question])

    # Calculate the cosine similarity of the user's question
    question_similarity = cosine_similarity(question, X)

    # Get the index of the most similar intent
    most_similar_index = question_similarity.argmax()

    # Return the corresponding answer
    answer = intents["intents"][most_similar_index]["answer"]
    if answer == "I'm not sure how to help with that. Let me try to find an answer with GPT-2":
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
        input_ids = gpt2_tokenizer.encode(question, return_tensors="pt")
        outputs = gpt2_model.generate(input_ids, do_sample=True, max_length=50)
        return gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        return answer
