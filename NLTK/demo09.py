# text classification
# moving review dataset
import nltk
import random
from nltk.corpus import movie_reviews

# documents = [(list(movie_reviews.words(fileid)),category)
#              for category in movie_reviews.categories()
#              for fileid in movie_reviews.fileids(category)]
documents = []
for category in movie_reviews.categories():     # movie_reviews.categories() = ["neg","pos"]
    for fileid in movie_reviews.fileids(category):      # movie_reviews.fileids(category) = files name
        documents.append((list(movie_reviews.words(fileid)),category))      # list(movie_reviews.words(fileid) : all contents in that file

random.shuffle(documents)
# print(documents[0])

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
print(all_words.most_common(15))    # the most 15 frequent words
print(all_words["stupid"])  # the frequency of the word 'stupid'