# lemmatizing
# A very similar operation to stemming is called lemmatizing.
# The major difference between these is, as you saw earlier,
# stemming can often create non-existent words.

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("cats"))

print(lemmatizer.lemmatize("better"))
print(lemmatizer.lemmatize("better",pos="a"))   # a: adjective
print(lemmatizer.lemmatize("best",pos="a"))
print(lemmatizer.lemmatize("run"))
print(lemmatizer.lemmatize("run","v"))  # v: verb