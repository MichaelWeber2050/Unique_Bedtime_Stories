#!/usr/bin/env python
# coding: utf-8

# ## Raw text file to predictive story teller

# In[ ]:


# dependecies on these packages and libraries
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import sys

import re
import string

import keras.models
import pickle


# __data__

# In[ ]:


# load in the txt data and preview 
dirty_text=(open('your_file_name').read())
dirty_text=dirty_text.lower()
dirty_text[0:2500]


# __pre processing__

# In[ ]:


# disgusting cleaning one by one 

# need to re write 
# remove double spaces and line endings to avoid counting

dirty_text = dirty_text.replace("'", "")
dirty_text = dirty_text.replace("-", "")
dirty_text = dirty_text.replace(":", "")
dirty_text = dirty_text.replace("2", "")
dirty_text = dirty_text.replace("0", "")
dirty_text = dirty_text.replace("3", "")
dirty_text = dirty_text.replace("[", "")
dirty_text = dirty_text.replace("#", "")
dirty_text = dirty_text.replace("1", "")
dirty_text = dirty_text.replace("5", "")
dirty_text = dirty_text.replace("]", "")
dirty_text = dirty_text.replace("6", "")
dirty_text = dirty_text.replace("8", "")
dirty_text = dirty_text.replace("*", "")
dirty_text = dirty_text.replace("’", "")
dirty_text = dirty_text.replace("9", "")
dirty_text = dirty_text.replace("4", "")
dirty_text = dirty_text.replace("—", "")
dirty_text = dirty_text.replace("_", "")
dirty_text = dirty_text.replace("ü", "u")
dirty_text = dirty_text.replace("(", "")
dirty_text = dirty_text.replace("î", "i")
dirty_text = dirty_text.replace("ô", "o")
dirty_text = dirty_text.replace(")", "")



dirty_text = dirty_text.replace("\n", " ")
dirty_text = dirty_text.replace("  ", " ")

dirty_text = dirty_text.replace(" ,  ", ", ")
dirty_text = dirty_text.replace("www.gutenberg.org.", "")
dirty_text = dirty_text.replace(".", "")
dirty_text = dirty_text.replace(",", "")
dirty_text = dirty_text.replace("!", "")
dirty_text = dirty_text.replace('"', "")


dirty_text = dirty_text.replace("  ", " ")
dirty_text = dirty_text.replace("“", "")

dirty_text = dirty_text.replace("”", "")
dirty_text = dirty_text.replace(";", "")


# In[ ]:


# look back at it
clean_text = dirty_text
clean_text[0:3000]


# __sort and map the characters__

# In[ ]:


# sort the unique characters that appear
_characters = sorted(list(set(clean_text)))
# map the unique characters to a dictionary with char as key and len of set list as value
_n_to_char = {n:char for n, char in enumerate(_characters)}
# map the unique characters to a dictionary with len of set list as key and char as value
_char_to_n = {char:n for n, char in enumerate(_characters)}


# __create lists of characters__

# In[ ]:


# create lists of characters
_X = []
_Y = []
length = len(clean_text)
seq_length = 100
for i in range(0, length-seq_length, 1):
    sequence = clean_text[i:i + seq_length]
    label = clean_text[i + seq_length]
    _X.append([_char_to_n[char] for char in sequence])
    _Y.append(_char_to_n[label])


# __modify the lists for model input__

# In[ ]:


unique_characters = set(clean_text)
num_classes = len(unique_characters)


# In[ ]:


_X_modified = np.reshape(_X, (len(_X), seq_length, 1))
# normalize the X data
_X_modified = _X_modified / float(len(_characters))
# one hot encode the output Y variable 
_Y_modified = to_categorical(_Y, num_classes=41)


# __create the model__

# In[ ]:


_model_ = Sequential()
_model_.add(LSTM(700, input_shape=(_X_modified.shape[1], _X_modified.shape[2]), 
               return_sequences=True))
_model_.add(Dropout(0.2))
_model_.add(LSTM(700))
_model_.add(Dropout(0.2))
_model_.add(Dense(_Y_modified.shape[1], activation='softmax'))
_model_.compile(loss='categorical_crossentropy', optimizer='adam')


# __add checkpoints to capture model weights at each epoch__

# In[ ]:


# define the checkpoint, do this before fitting but not needed unless you intend 
# to fit the model below

filepath="model-weights-{epoch:02d}-{loss:.4f}.file"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, 
                              save_best_only=True, mode='min')
callbacks_list = [checkpoint]


# __fit the model__
# 
# *** this could require a GPU

# In[ ]:


# suggest using a GPU to fit this model
_model_.fit(_X_modified, _Y_modified, 
            epochs=17, 
            batch_size=100, 
            callbacks=callbacks_list)


# __load in the saved trained model__

# In[ ]:


# load the network weights
filename = "model-weights-EPOCH-LOSS.file"
_model_.load_weights(filename)
_model_.compile(loss='categorical_crossentropy', optimizer='adam')


# __write some text__

# In[ ]:


# print out some text :)

n_vocab = len(_characters)

start = np.random.randint(0, len(_X)-1)
pattern = _X[start]
#print("Seed:")
#print("\"", ''.join([_n_to_char[value] for value in pattern]), "\"")


# write some words hopefully they make sense 
for i in range(400):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = _model_.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = _n_to_char[index]
    seq_in = [_n_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print("\n\n here is your story enjoy :)")

