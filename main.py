from flask import Flask, request, jsonify
from intents_preprocessing import generate_response, intents
from model_prediction import classify_intent

app = Flask(__name__)

@app.route('/question', methods=['POST'])
def question():
    question = request.get_json()['question']
    intent_index = classify_intent(question)
    if intent_index in range(len(intents["intents"])):
        answer = intents["intents"][intent_index]["answer"]
    else:
        answer = generate_response(question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run()
