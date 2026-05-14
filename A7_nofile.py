# pip install nltk

import nltk
from nltk.tokenize import *
from nltk.corpus import *
from nltk.stem import *
import re

# Download Required Data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

text = "It is the process of splitting a text document into individual sentences based on punctuation marks"

# Sentence Tokenization
var1 = sent_tokenize(text)
print("Sentence Tokenization:")
print(var1)

# Word Tokenization
var2 = word_tokenize(text)
print("\nWord Tokenization:")
print(var2)

# Removing Punctuation
text = re.sub('[^a-zA-Z]', ' ', text)
print("\nAfter Removing Punctuation:")
print(text)

# Removing Stopwords
var3 = set(stopwords.words('english'))
tokens = word_tokenize(text.lower())
filtered_text = []
for word in tokens:
    if word not in var3:
        filtered_text.append(word)
print("\nFiltered Sentence:")
print(filtered_text)

# Stemming
print("\nStemming:")
words = ["write", "writing", "wrote", "writes", "reading", "reads"]
ps = PorterStemmer()
for w in words:
    print(ps.stem(w))

# Lemmatization
print("\nLemmatization:")
wordnet_lemmatizer = WordNetLemmatizer()
text2 = "studies studying cries cry"
tt = nltk.word_tokenize(text2)
for w in tt:
    print("Lemma for {} is {}".format(
        w,
        wordnet_lemmatizer.lemmatize(w)
    ))

# POS Tagging
print("\nPOS Tagging:")
text3 = "Hello everyone this is a sample text Earth"
text3 = nltk.word_tokenize(text3)
print(nltk.pos_tag(text3))

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
new_sentence = [
    "This is an example of term frequency meow meow meow"
]
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(new_sentence)
print("\nTF-IDF Matrix:")
print(tfidf.toarray())
print("\nFeature Names:")
print(vectorizer.get_feature_names_out())