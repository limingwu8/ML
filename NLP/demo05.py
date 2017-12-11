# Name Entity Recognition
# Named entity recognition is useful to quickly find out what the subjects of discussion are.
# NLP can either recognize a general named entity, or it can even recognize locations, names,
# monetary amounts, dates, and more.

# part of speech tagging
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

# PunktSentenceTokenizer is an sentence boundary detection algorithm that must be trained to be used
# PunktSentenceTokenizer is the abstract class for the default sentence tokenizer
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            namedEnt = nltk.ne_chunk(tagged,binary=True)
            namedEnt.draw()

    except Exception as e:
        print(str(e))

process_content()