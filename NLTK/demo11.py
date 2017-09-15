
# text classification
# save and read classifier by using Pickle
import nltk
import random
from nltk.corpus import movie_reviews
import pickle

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

all_words = nltk.FreqDist(all_words)    # nltk.FreqDist : like a dictionary, key are words, values are the frequency of that word.

word_features = list(all_words.keys())[:3000]   # get top frequent 3000 words and regard these words as features.

def find_features(document):
    words = set(document)   # convert the content in one file to a set.
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

# print(find_features(movie_reviews.words('neg/cv000_29416.txt')))    # find all feature words in this txt file.
# featuresets : [({'plot':False,':':False...},'pos'),(),()...]
featuresets = [(find_features(rev),category) for (rev,category) in documents]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# classifier = nltk.NaiveBayesClassifier.train(training_set)

# save classifier to file
# save_classifier = open("naivebayes.pickle","wb")
# pickle.dump(classifier,save_classifier)
# save_classifier.close()

# load classifier from file
classifier_f = open("naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

# test classifier
print("Naive Bayes Algo accuracy percent:",(nltk.classify.accuracy(classifier,testing_set))*100)
classifier.show_most_informative_features(15)