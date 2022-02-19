import numpy as np
import pickle
from keras.models import load_model
hans_model = load_model('simple_hans.hd5')
hans_model._make_predict_function()
hans_n_to_char = pickle.load(open("hans_n_to_char.pkl", "rb"))
hans_X = pickle.load(open("hans_X.pkl", "rb"))

def make_story():

    n_vocab = 43# len(hans_characters)

    start = np.random.randint(0, len(hans_X)-1) # len(hans_X)-1
    pattern = hans_X[start]
    initial_text = ''.join([hans_n_to_char[value] for value in pattern])
    print("Seed:")
    print(f'"{initial_text}"')
    # generate characters
    char_list = []
    for i in range(300):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = hans_model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = hans_n_to_char[index]
        seq_in = [hans_n_to_char[value] for value in pattern]
        char_list.append(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print("\nDone.")

    return initial_text, ''.join(char_list)
