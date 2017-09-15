# split string to sentences or words

from nltk.tokenize import sent_tokenize, word_tokenize

# how to split the string to sentences? you can write regular expressions, but NLTK can do it better.
example_text = ("hello Mr. Smith, how are you doing today? The weather is great and Python is awesome."
            " The sky is pinkish-blue. You should not eat cardboard.")
print(sent_tokenize(example_text))  # split to sentences
print(word_tokenize(example_text))  # split to words

for i in sent_tokenize(example_text):
    print(i)