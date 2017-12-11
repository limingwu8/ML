# e.g.
# lexicon array: [chair,table,spoon,television]
# sentence: I pulled the chair up to the table
# np.zeros(len(lexicon)), [0,0,0,0]
# result for this sentence: [1 1 0 0]

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer
hm_lines = 10000000     # total lines
