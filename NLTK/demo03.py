# stemming
# e.g. stemming will convert ["python","pythoner","pythoning","pythoned","pythonly"] to python
# e.g. stemming will convert ["interesting","interested"] to interest
# stemming may create some words that do not exits
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()
example_words = ["python","pythoner","pythoning","pythoned","pythonly"]

for w in example_words:
    print(ps.stem(w))

new_text = ("It is very import to be pythonly while you are pythoning with python. "
            "All pythoners have have pythoned poorly at least once.")
words = word_tokenize(new_text)
for w in words:
    print(ps.stem(w))