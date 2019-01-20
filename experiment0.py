#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 09:33:40 2019

@author: amc
"""

 # -------------------- script for A.I. -----------------------#
import numpy
import pandas
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.snowball import SnowballStemmer


ps = SnowballStemmer('english')

def preprocess(text):
            # Stem and remove stopwords
            text = re.sub('[^a-zA-Z]', ' ', text)
            text = text.lower()
            text = text.split()
            text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
            return ' '.join(text)
        
dataset1 = pandas.read_csv('interview1.csv', encoding='ISO-8859-1')
dataset2 = pandas.read_csv('interview2.csv', encoding='ISO-8859-1')
dataset3 = pandas.read_csv('interview3.csv', encoding='ISO-8859-1')

dataset = pandas.concat([dataset1, dataset2, dataset3])
#Reset index otherwise during the loop below we select multiple rows (pandas.concat results in repeated indices)
dataset = dataset.reset_index(drop=True)

querycorpus = []
for i in range(0, len(dataset)):
    query = re.sub('[^a-zA-Z]', ' ', dataset['Q'][i])
    query = query.lower()
    query = query.split()
    query = [ps.stem(word) for word in query if not word in set(stopwords.words('english'))]
    query = ' '.join(query)
    querycorpus.append(query)      

# Creating the Bag of Words model with TFIDF and calc cosine_similarity
vectorizer = CountVectorizer(decode_error="replace")
vec_train = vectorizer.fit_transform(querycorpus) #this is needed to get the attribute vocabulary_
training_vocabulary = vectorizer.vocabulary_
transformer = TfidfTransformer()
trainingvoc_vectorizer = CountVectorizer(decode_error="replace", vocabulary=training_vocabulary)
tfidf_querycorpus = TfidfVectorizer().fit_transform(querycorpus)



def answer(newquery, k):

    tfidf_newquery = transformer.fit_transform(trainingvoc_vectorizer.fit_transform(numpy.array([preprocess(newquery)]))) 
    cosine_similarities = cosine_similarity(tfidf_newquery, tfidf_querycorpus)
    related_docs_indices = (-cosine_similarities[0]).argsort()
    sorted_freq = cosine_similarities[0][related_docs_indices]
    
    #note for this distance the problem we had befor with inf, we have now with 0. Again we decide
    #to make the prediction a bit random. This could be adjusted to remove any 0 distance and
    #pick the only ones left if any, and if none predict 1.
    
    if sum(sorted_freq)==0:
        return "Not understood"
    
    elif sorted_freq[k-1]!=sorted_freq[k] or sorted_freq[k-1]==sorted_freq[k]==0:
        selected = related_docs_indices[:k]
       
#        return dataset.iloc[selected[0]]['A']
        return dataset.iloc[selected[0]]['A'], dataset.iloc[selected,:(k-1)]   
        print("\n Cosine Similarities:", sorted_freq[:k], "\n")
    else:
        indeces = numpy.where(numpy.roll(sorted_freq,1)!=sorted_freq)
        selected = related_docs_indices[:indeces[0][indeces[0]>=k][0]]
    
#        return dataset.iloc[selected[0]]['A']
        return dataset.iloc[selected[0]]['A'], dataset.iloc[selected,:(k-1)]
        print("\n Cosine Similarities:", sorted_freq[:k], "\n")