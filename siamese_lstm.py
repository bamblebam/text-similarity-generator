# %%
from time import time
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
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
# %%
EMBEDDING_FILE = 'embeddings_google/GoogleNews-vectors-negative300.bin'
EMBEDDING_FILE2 = 'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz'
# %%
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
# %%
