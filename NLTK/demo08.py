# Part of the NLTK Corpora is WordNet.
# I would not totally classify WordNet as a Corpora,
# if anything it is really a giant Lexicon, but, either way, it is super useful.
# With WordNet we can do things like look up words and their meaning according to their parts of speech,
# we can find synonyms, antonyms, and even examples of the word in use.

from nltk.corpus import wordnet

syns = wordnet.synsets("program")

# print(syns)
# # synset
# print(syns[0].name())
# # definition
# print(syns[0].definition())
# # examples
# print(syns[0].examples())

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")
print(w1.wup_similarity(w2))

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("car.n.01")
print(w1.wup_similarity(w2))

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("cat.n.01")
print(w1.wup_similarity(w2))