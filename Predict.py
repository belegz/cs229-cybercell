import numpy as np
import random as rn

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/fchollet/keras/issues/2280#issuecomment-306959926

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(421)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(123451)

import logging

from keras import regularizers
from keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTM, TimeDistributed
from keras.models import Sequential, load_model

from data.datasets import *
from eval import keras_metrics, metrics
from nlp import tokenizer as tk
from utils import info, preprocessing, postprocessing, plots

# LOGGING CONFIGURATION

#logging.basicConfig(
#    format='%(asctime)s\t%(levelname)s\t%(message)s',
#    level=logging.DEBUG)

#info.log_versions()

# END LOGGING CONFIGURATION

# GLOBAL VARIABLES

SAVE_MODEL = False
MODEL_PATH = "models/simplernn.h5"
SHOW_PLOTS = False

# END GLOBAL VARIABLES

# Dataset and hyperparameters for each dataset

#DATASET = Hulth
DATASET = Datasetnew

if DATASET == Datasetnew:
    tokenizer = tk.tokenizers.nltk
    DATASET_FOLDER = "data/Datasetnew"
    MAX_DOCUMENT_LENGTH = 550
    MAX_VOCABULARY_SIZE = 20000
    EMBEDDINGS_SIZE = 300
    BATCH_SIZE = 32
    EPOCHS = 200
    KP_WEIGHT = 10
    STEM_MODE = metrics.stemMode.none
    STEM_TEST = False
else:
    raise NotImplementedError("Can't set the hyperparameters: unknown dataset")

# END PARAMETERS

#logging.info("Loading dataset...")

def predict_command(path,com_list):

#    data = DATASET(DATASET_FOLDER)
    data = DATASET(path)

    test_doc_str, test_answer_str = data.load_test()
#    test_doc_str = path
#    print("test_doc_str",test_doc_str)

    test_doc, test_answer = tk.tokenize_set(test_doc_str, test_answer_str, tokenizer)
#    test_doc = tk.tokenize_set_test(test_doc_str, tokenizer)
#    print("test_doc",test_doc)

    test_x, embedding_matrix = preprocessing. \
        prepare_sequential_test(test_doc, test_answer,
                           max_document_length=MAX_DOCUMENT_LENGTH,
                           max_vocabulary_size=MAX_VOCABULARY_SIZE,
                           embeddings_size=EMBEDDINGS_SIZE,
                           stem_test=STEM_TEST)

# weigh training examples: everything that's not class 0 (not kp)
# gets a heavier score
    from sklearn.utils import class_weight

    model = load_model(MODEL_PATH)

    output = model.predict(x=test_x, verbose=0)

    obtained_tokens = postprocessing.undo_sequential(output)
    obtained_words = postprocessing.get_words(test_doc, obtained_tokens)
    # print("obtained_words",obtained_words)
    clean_words = postprocessing.get_valid_patterns(obtained_words)
    # print("clean_words",clean_words)
    obtained_words_top = postprocessing.get_top_words(test_doc, output, 5)
    # print("obtained_words_top",obtained_words_top)

    msg = []
    for i in range(len(com_list)):
        msg.append(obtained_words[com_list[i]])

    return msg
#predict_command(DATASET_FOLDER)