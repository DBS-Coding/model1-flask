from flask import Flask, request, jsonify, render_template
import numpy as np
import string
import random
import pickle
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the saved model and necessary preprocessing objects
model = load_model('chatbot_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_encoder.pickle', 'rb') as enc:
    le = pickle.load(enc)
with open('responses.pickle', 'rb') as resp:
    responses = pickle.load(resp)

# Define the input shape used during training
input_shape = 20  # Update this based on your actual input shape

@app.route('/')
def home():
    return "API Histotalk Model 1.0"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    user_input = data['message']
    
    # Text preprocessing (same as in your training code)
    text_p = []
    user_input = [letters.lower() for letters in user_input if letters not in string.punctuation]
    user_input = ''.join(user_input)
    text_p.append(user_input)
    
    # V1 - Error shape -------
    # Tokenizing and padding
    # user_input = tokenizer.texts_to_sequences(text_p)
    # user_input = np.array(user_input).reshape(-1)
    # user_input = pad_sequences([user_input], input_shape)
    # End V1 ----
    
    # V2 - Working ----
    # Tokenizing
    sequences = tokenizer.texts_to_sequences(text_p)
    
    # Instead of reshaping and padding as before, we need to adjust to match (None, 3)
    # If this is a sequence model and we need exactly 3 tokens
    padded_sequences = pad_sequences(sequences, maxlen=3, padding='post', truncating='post')
    # End V2 ----
    
    
    # Log untuk debugging
    print(f"\nInput shape after padding: {padded_sequences.shape}\n")
    
    ## v1 original test ---- error ☠️ input shape
    # Getting prediction from model
    # output = model.predict(user_input)
    # End v1 ----
    
    ## v2 original test ---- Working ✅ Correct Shape
    # Getting prediction from model
    output = model.predict(padded_sequences)
    # End v2 ----
    
    ## dummy test --- Working ✅ Correct Shape
    # dummy_input = np.random.random((1, 3))
    # output = model.predict(dummy_input)
    
    output = output.argmax()
    
    # Finding the right tag and response
    response_tag = le.inverse_transform([output])[0]
    bot_response = random.choice(responses[response_tag])
    
    return jsonify({'response': bot_response, 'tag': response_tag})

if __name__ == '__main__':
    app.run(debug=True)