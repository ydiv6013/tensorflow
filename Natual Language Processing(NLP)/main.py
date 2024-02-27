import nltk
from nltk import word_tokenize,sent_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

# Tokenisation : Tokenization refers to break down the text into smaller units.
paragraph = "Welcome to the captivating world of Multi-Modal Deep Learning, where images and text intertwine to unlock the true potential of Artificial Intelligence! In this article, we embark on a journey that combines the prowess of computer vision and natural language processing (NLP) to tackle complex tasks that involve understanding and processing information from diverse modalities."

word_tokenised = nltk.word_tokenize(paragraph)
sent_tokenize = nltk.sent_tokenize(paragraph)

print(word_tokenised)
print(sent_tokenize)

# Canonicalization : is a process of mapping a base word.
# plays,played,playing,play conveys the same action play.  base word is "play"
# stemming  removes the affix from the base word using certain set of rules.

# create an object of class PorterStemmer

porter = PorterStemmer()
print(porter.stem("play"))
print(porter.stem("Learning"))
print(porter.stem("played"))
print(porter.stem("plays"))
print(porter.stem("Communication"))

# Lemmatization
"""param pos: The Part Of Speech tag. Valid options are "n" for nouns,
"v" for verbs, "a" for adjectives, "r" for adverbs and "s" for satellite adjectives."""
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("Learning","v"))
print(lemmatizer.lemmatize("Coding","v"))
print(lemmatizer.lemmatize("Communication","v"))

# Part Of Speech tagging (POS) :  refers to assigning each word of a sentence to its part of speech.
tags = pos_tag(word_tokenised)
print(tags)
