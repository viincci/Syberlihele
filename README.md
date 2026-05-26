# Syberlihele
## Project Overview
Syberlihele is a conversational AI project designed to provide assistance with brand strategy and management. It utilizes natural language processing (NLP) and machine learning algorithms to understand user queries and generate relevant responses.
## Key Features
- Intent classification and response generation
- Integration with GPT-2 for answering unknown questions
- Support for multiple intents and entities
- Customizable and extensible architecture
## Tech Stack
- Python 3.x
- Flask for web development
- TensorFlow and Keras for machine learning
- NLTK and spaCy for NLP tasks
- GPT-2 for text generation
## Installation
1. Clone the repository: `git clone https://github.com/username/Syberlihele.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Train the model: `python train.py`
4. Run the application: `python main.py`
## Usage
1. Send a POST request to `http://localhost:5000/question` with a JSON body containing the user\'s question
2. The application will respond with a JSON object containing the generated answer
## Environment Variables
- `ACTION_ENDPOINT`: the URL of the action endpoint
- `OPENAI_API_KEY`: the API key for OpenAI\'s GPT-2 model
## Code
```python
# example code snippet
from flask import Flask, request, jsonify
app = Flask(__name__)
@app.route('/question', methods=['POST'])
def question():
    # handle user question and generate response
    pass
```
Note: This is a basic example and may require modifications to work with your specific use case.