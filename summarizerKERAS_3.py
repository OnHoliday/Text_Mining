# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 15:09:30 2019

@author: Konrad
"""

import pandas as pd
import re
from nltk.corpus import stopwords
from pickle import dump, load

reviews = pd.read_csv("Amazon/Reviews.csv")
print(reviews.shape)
print(reviews.head())
print(reviews.isnull().sum())

reviews = reviews.dropna()
reviews = reviews.drop(['Id','ProductId','UserId','ProfileName','HelpfulnessNumerator','HelpfulnessDenominator', 'Score','Time'], 1)
reviews = reviews.reset_index(drop=True) 
print(reviews.head())

for i in range(5):
    print("Review #",i+1)
    print(reviews.Summary[i])
    print(reviews.Text[i])            
    print()
    
contractions = {"ain't": "am not","aren't": "are not","can't": "cannot","can't've": "cannot have",
"'cause": "because","could've": "could have","couldn't": "could not","couldn't've": "could not have","didn't": "did not","doesn't": "does not",
"don't": "do not","hadn't": "had not","hadn't've": "had not have","hasn't": "has not","haven't": "have not","he'd": "he would","he'd've": "he would have"}

def clean_text(text, remove_stopwords=True):
    # Convert words to lower case
    text = text.lower() 
    if True: 
        text = text.split()     
        new_text = []    
        for word in text:    
            if word in contractions:    
                new_text.append(contractions[word])    
            else:    
                new_text.append(word)                
                text = " ".join(new_text)                
                text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)                
                text = re.sub(r'\<a href', ' ', text)                
                text = re.sub(r'&amp;', '', text)                
                text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)                
                text = re.sub(r'<br />', ' ', text)                
                text = re.sub(r'\'', ' ', text)    
    if remove_stopwords:    
        text = text.split()        
        stops = set(stopwords.words("english"))        
        text = [w for w in text if not w in stops]        
        text = " ".join(text)    
    return text

import nltk
nltk.download('stopwords')

# Clean the summaries and texts
reviews = reviews[:1000]

clean_summaries = []
for summary in reviews.Summary:
    clean_summaries.append(clean_text(summary, remove_stopwords=False))

print("Summaries are complete.")

clean_texts = []
for text in reviews.Text:
    clean_texts.append(clean_text(text))
print("Texts are complete.")

stories = list()

for i, text in enumerate(clean_texts):
    stories.append({'story': text, 'highlights': clean_summaries[i]})

# save to file

dump(stories, open('Amazon/review_dataset.pkl', 'wb'))


batch_size = 64
epochs = 5
latent_dim = 256
num_samples = 1000
stories = load(open('Amazon/review_dataset.pkl', 'rb'))

print('Loaded Stories %d' % len(stories))
print(type(stories))

input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

for story in stories:
    input_text = story['story']

for highlight in story['highlights']:
    target_text = highlight
    target_text = '\t' + target_text + '\n' # We use "tab" as the "start sequence" character for the targets, and "\n" as "end sequence" character.
    input_texts.append(input_text)
    target_texts.append(target_text)

    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

#################################


from keras.models import Model
from keras.layers import Input, Dense, Reshape, merge, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence
import tensorflow as tf

#def define_models(n_input, n_output, n_units):

    # define training encoder
encoder_inputs = Input(shape=(None, n_input))    
encoder1 = Embedding(vocab_size, 128)(encoder_inputs)
encoder2 = LSTM(n_units)(encoder1)  
encoder3 = RepeatVector(sum_txt_length)(encoder2)

# define training decoder    
decoder1 = LSTM(128, return_sequences=True)(encoder3)   
outputs = TimeDistributed(Dense(vocab_size, activation='softmax'))(decoder1)

model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')    
        # return all models    
#    return model, encoder_model, decoder_model


#model, infenc, infdec = define_models(31, 11, 128)

#model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size,

epochs=epochs,

validation_split=0.2)

# Save model

model.save('/deeplearning-keras/ch09/summarization/model2.h5')

###########################################################

from keras.layers import RepeatVector, TimeDistributed, LSTM


from keras.models import Model
from keras.layers import Input, Dense, Reshape, merge
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence

from keras.layers import concatenate, dot
from keras.layers import Dot

import urllib
import collections
import os
import zipfile

import numpy as np
import tensorflow as tf

from utils_ import *
import pandas as pd
from nltk.tokenize import RegexpTokenizer

database = r'articles.db'
sql = SQL(database)
x = sql.get_sample(5)

tokenizer = RegexpTokenizer(r'\w+')

input_to_wordVec = []
input_to_wordVec_title = []
for i, value in x.iterrows():
    input_to_wordVec.append(tokenizer.tokenize(value['content']))
    input_to_wordVec_title.append(tokenizer.tokenize(value['title']))

flatten = [word for i in input_to_wordVec for word in i]   
flatten2 = [word for i in input_to_wordVec_title for word in i]   
flatten_all = flatten + flatten2
def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

def collect_data():
#    url = 'http://mattmahoney.net/dc/'
#    filename = r'C:\Users\Konrad\Desktop\NOVA IMS\text mining\text8.zip' # maybe_download('text8.zip', url, 31344016)
    vocabulary = flatten_all #   read_data(filename)
    
    vocabulary_size = len(flatten_all)
    print(vocabulary[:7])
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                                vocabulary_size)
    del vocabulary  # Hint to reduce memory.
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = collect_data()


input_list = []
for i in input_to_wordVec:
    temp_list = [] 
    for word in i:
        temp_list.append(dictionary[word])
    temp_list = temp_list[:100]
    input_list.append(temp_list)

input_list_title = []
for i in input_to_wordVec_title:
    temp_list = [] 
    for word in i:
        temp_list.append(dictionary[word])
    temp_list = temp_list[:12]
    input_list_title.append(temp_list)

X = input_list
y = input_list_title

vocab_size = len(dictionary)
src_txt_length = 100
sum_txt_length = 12
# encoder input model
inputs = Input(shape=(src_txt_length,))
encoder1 = Embedding(vocab_size, 128)(inputs)
encoder2 = LSTM(128)(encoder1)
encoder3 = RepeatVector(sum_txt_length)(encoder2)
# decoder output model
decoder1 = LSTM(128, return_sequences=True)(encoder3)
outputs = TimeDistributed(Dense(vocab_size, activation='softmax'))(decoder1)
# tie it together
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(X, y, epochs=6, batch_size=16)





















###################################### predicting
# generate target given source sequence

def predict_sequence(infenc, infdec, source, n_steps, cardinality):

# encode

state = infenc.predict(source)

# start of sequence input

target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)

# collect predictions

output = list()

for t in range(n_steps):

# predict next char

yhat, h, c = infdec.predict([target_seq] + state)

# store prediction

output.append(yhat[0,0,:])

# update state

state = [h, c]

# update target sequence

target_seq = yhat

return array(output)