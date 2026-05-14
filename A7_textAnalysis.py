"""
Text Analytics
1. Extract Sample document and apply following document preprocessing methods:
Tokenization, POS Tagging, stop words removal, Stemming and Lemmatization.
2. Create representation of document by calculating Term Frequency and Inverse Document
Frequency(TF-IDF).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk #NLTK is a leading platform for building Python programs to work with human language data

file=open('simple.txt','r')
print(file.read())

nltk.download('all')

file.seek(0)
content=file.read()
content

import nltk
print(nltk.data.path)

import nltk

nltk.download("gutenberg") #(collection of books) that NLTK provides.

from nltk.corpus import gutenberg

text = gutenberg.raw("shakespeare-hamlet.txt")  # full text as one big string.
small_text = " ".join(sent_tokenize(text)[:5])

#encoding="utf-8" supports all characters safely
with open("simple.txt", "w", encoding="utf-8") as f:
    f.write(small_text)

print("Hamlet dataset subset saved into simple.txt")

file.seek(0)
content=file.read() #string
content

from nltk.tokenize import sent_tokenize

sentence=sent_tokenize(content)
sentence

#custom tokenize
from nltk.tokenize import RegexpTokenizer
tokenizer=RegexpTokenizer(f"\w+")
words=tokenizer.tokenize(content)
words

#POS tagging and stop word removing
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stopWords=set(stopwords.words('english'))

print(stopWords)

for sen in sentence:
    Words=word_tokenize(sen)
    filteredWords=[word.lower() for word in Words if word.lower() not in stopWords]
    print(f"words without stopwords{filteredWords}")
    print(f"words with stopwords{Words}")
    print(f"POS Tagging{nltk.pos_tag(filteredWords)}")

print(f"POS Tagging{nltk.pos_tag(filteredWords)}")  #last sentence

#stemming and lemmatization
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

#stemming - reduces words to their root/base form by removing suffixes.
stemmer=PorterStemmer() #obj
for word in Words:
    print(f"{word}- After Stemming = {stemmer.stem(word)}")

#Lemmatization - Lemmatization converts word into meaningful dictionary form (lemma).
lemmatizer=WordNetLemmatizer()
for word in Words:
    print(f"{word}:{lemmatizer.lemmatize(word)}")

#IF-IDF - TF-IDF converts text into numerical values based on word importance.
#TF (Term Frequency) How frequently a word appears in a document. IDF (Inverse Document Frequency) Reduces importance of very common words.
sentence=sentence[:3]   #first 3 sentence
new_sentence=[''.join(sentence)]
new_sentence    #list containing single string

from sklearn.feature_extraction.text import TfidfVectorizer

def calculate_tfIdf(document):
    tokenizer=TfidfVectorizer()
    tf_matrix=tokenizer.fit_transform(document)         #numeric representation of words
    features_names=tokenizer.get_feature_names_out()    #Extracts all unique words
    return tf_matrix,features_names

document=new_sentence

tf_matrix,feature_names=calculate_tfIdf(new_sentence)
print('TFIDF')
feature_names,tf_matrix.toarray()

