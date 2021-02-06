# %%
from time import time
from keras import optimizers
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

import itertools
import datetime

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Embedding, LSTM, Lambda, Dense, concatenate, add, subtract
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.layers.merge import concatenate
import tensorflow as tf
from tensorflow.python.client import device_lib
import pickle as pickle
# %%
EMBEDDING_FILE = 'embeddings_google/GoogleNews-vectors-negative300.bin'
EMBEDDING_FILE2 = 'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz'
# %%
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
# %%
TRAIN_PATH = "./datasets/train.csv"
TEST_PATH = "./datasets/test.csv"
# %%
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
train_df.head()
test_df.head()
# %%
# train_df = train_df.sample(frac=0.2, random_state=42)
# test_df = test_df.sample(frac=0.2, random_state=42)
# %%
stop = set(stopwords.words("english"))
# %%


def text_to_word_list(text):
    ''' Pre process and convert texts to a list of words '''
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.split()
    return text


# %%
text = "Hello my name is loca and i am a little pug"
test = text_to_word_list(text)
# %%
question_cols = ["question1", "question2"]
dfs = [train_df, test_df]
# %%
vocabulary_dict = dict()
inverse_vocabulary = ["<unk>"]
# %%
for df in dfs:
    for index, row in df.iterrows():
        for question in question_cols:
            ques2num = list()
            for word in text_to_word_list(row[question]):
                if word in stop and word not in word2vec.vocab:
                    continue
                if word not in vocabulary_dict:
                    vocabulary_dict[word] = len(inverse_vocabulary)
                    ques2num.append(len(inverse_vocabulary))
                    inverse_vocabulary.append(word)
                else:
                    ques2num.append(vocabulary_dict[word])
            df.at[index, question] = ques2num

# %%
train_df.head()
# %%
embedding_dims = 300
embeddings = 1*np.random.randn(len(vocabulary_dict)+1, embedding_dims)
embeddings[0] = 0
# %%
for word, index in vocabulary_dict.items():
    if word in word2vec.vocab:
        embeddings[index] = word2vec.word_vec(word)
# %%
# train_df.to_csv("./datasets/new_train.csv")
# test_df.to_csv("./datasets/new_test.csv")
# %%
max_seq_len = 100
X = train_df[question_cols]
Y = train_df["is_duplicate"]
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1)
# %%
X_train = {'left': X_train.question1, 'right': X_train.question2}
X_val = {'left': X_val.question1, 'right': X_val.question2}
X_test = {'left': test_df.question1, 'right': test_df.question2}
Y_train = Y_train.values
Y_val = Y_val.values
# %%
for df, side in itertools.product([X_train, X_val], ['left', 'right']):
    df[side] = pad_sequences(df[side], maxlen=max_seq_len)

# %%
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)
# %%


def exponent_negative_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))


# %%
left_input = Input(shape=(max_seq_len,), dtype='int32')
right_input = Input(shape=(max_seq_len,), dtype='int32')
embedding_layer = Embedding(len(embeddings), embedding_dims, weights=[
                            embeddings], input_length=max_seq_len, trainable=False)

# %%
left_encoded = embedding_layer(left_input)
right_encoded = embedding_layer(right_input)

# %%
shared_lstm = LSTM(64)
left_output = shared_lstm(left_encoded)
right_output = shared_lstm(right_encoded)
# %%
malstm_distance = Lambda(function=lambda x: exponent_negative_manhattan_distance(
    x[0], x[1]), output_shape=lambda x: (x[0][0], 1))([left_output, right_output])
# %%
# diff = subtract([left_output, right_output])
# summation = add([left_output, right_output])
# concatenated_x = concatenate([left_output, right_output, diff, summation], 1)
# %%
# dense1 = Dense(64, activation='relu')(concatenated_x)
# dense2 = Dense(1, activation='sigmoid')(dense1)
# %%
malstm_model = Model([left_input, right_input], [malstm_distance])
optimizer = Adadelta(clipnorm=1.25)
malstm_model.compile(loss='mean_squared_error',
                     optimizer='adam', metrics=['accuracy'])

# %%
checkpoint = ModelCheckpoint("./models/model6.h5", save_best_only=True)
history = malstm_model.fit([X_train['left'], X_train['right']], Y_train, batch_size=1024, epochs=10, validation_data=([
    X_val['left'], X_val['right']], Y_val), callbacks=[checkpoint], shuffle=True)

# %%
new_model = load_model("./models/model1.h5")
# %%
pred = malstm_model.predict([X_train['left'], X_train['right']])
# %%
# print(X_train['right'][0])
# print(X_test['right'])
print(pred[90:100])
print(Y_train[90:100])

# %%
if tf.test.gpu_device_name():

    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
# %%
# with open("pickles/vocabulary_dict.pickle", "wb") as file1:
#     pickle.dump(vocabulary_dict, file1)
# with open("pickles/inverse_vocabulary.pickle", "wb") as file2:
#     pickle.dump(inverse_vocabulary, file2)
# with open("pickles/embeddings.pickle", "wb") as file3:
#     pickle.dump(embeddings, file3)

# %%
print(device_lib.list_local_devices())
# %%
