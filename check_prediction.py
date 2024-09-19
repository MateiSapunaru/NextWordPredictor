import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
import json

model = tf.keras.models.load_model('next_word_model.keras')

with open('tokenizer_string.txt', 'r') as f:
    tokenizer_json_string = f.read()
    tokenizer = tokenizer_from_json(json.loads(tokenizer_json_string))

max_sequence_len = model.input_shape[1] + 1

def generate_text(seed_text, next_words=1):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_word = tokenizer.index_word[np.argmax(predicted_probs)]
        seed_text += " " + predicted_word
    return seed_text

print("Press 'q' to stop generating text.")
seed_text = input("Enter your starting text: ")
while True:
    seed_text = generate_text(seed_text)
    print("Generated text:", seed_text)
    cont = input("Press Enter to continue or 'q' to quit: ").strip().lower()
    if cont == 'q':
        break
