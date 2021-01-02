import os
import json
import numpy as np
import pandas as pd

allLabels = []

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

import string

for filename in os.listdir("./dataset"):
    if filename.endswith(".json"):
        with open(f'./dataset/{filename}') as json_file:
            graph = json.load(json_file)
            features = graph['features']
            for key in features:
                label = features[key]
                allLabels.append(label)

X_train = np.array(allLabels)

def text_process(text):
    '''
    Takes in a string of text, then performs the following:
    3. Return the cleaned text as a list of words
    4. Remove words
    '''
    stemmer = WordNetLemmatizer()
    return [stemmer.lemmatize(word) for word in text]

#Vectorisation : -

from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconvert = TfidfVectorizer(analyzer=text_process,ngram_range=(1,3)).fit(X_train)
X_transformed = tfidfconvert.transform(X_train)

# Clustering the training sentences with K-means technique

from sklearn.cluster import KMeans
modelkmeans = KMeans(n_clusters=1, init='k-means++', n_init=100)
modelkmeans.fit(X_transformed)

print(modelkmeans.labels_)
print(X_train)

data = {'cluster': modelkmeans.labels_, 'label' : X_train}
df = pd.DataFrame(data)
print(df)
df.to_csv('relabelClusters.csv')

D = {}
for i in range(len(X_train)):
    D[X_train[i]] = str(modelkmeans.labels_[i])

for filename in os.listdir("./dataset"):
    if filename.endswith(".json"):
        graph = 1;
        with open(f'./dataset/{filename}') as json_file:
            graph = json.load(json_file)
            for key in graph['features']:
                value = graph['features'][key]
                graph['features'][key] = D[value]
        with open(f'./relabledDataset/{filename}', 'w') as json_file: 
            json.dump(graph, json_file)
