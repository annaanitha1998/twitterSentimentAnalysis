
import pandas as pd
import numpy as np
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import string 
from nltk.stem import WordNetLemmatizer
import pandas as pd
df1 = pd.read_csv("E:/final project 4th yr/Predicting-Cyberbulling-on-Twitter-master/test1.csv")
Tweet = []
Labels = []

for row in df1["Tweet"]:
    #tokenize words
    words = word_tokenize(row)
    #remove punctuations
    clean_words = [word.lower() for word in words if word not in set(string.punctuation)]
    #remove stop words
    english_stops = set(stopwords.words('english'))
    characters_to_remove = ["''",'``',"rt","https","’","“","”","\u200b","--","n't","'s","...","//t.c" ]
    clean_words = [word for word in clean_words if word not in english_stops]
    clean_words = [word for word in clean_words if word not in set(characters_to_remove)]
    #Lematise words
    wordnet_lemmatizer = WordNetLemmatizer()
    lemma_list = [wordnet_lemmatizer.lemmatize(word) for word in clean_words]
    Tweet.append(lemma_list)

    for row in df1["Text Label"]:
        Labels.append(row)
nltk.download('punkt')
combined = zip(Tweet, Labels)
nltk.download('stopwords')
nltk.download('wordnet')
combined = zip(Tweet, Labels)
def bag_of_words(words):
    return dict([(word, True) for word in words])
Final_Data = []
for r, v in combined:
    bag_of_words(r)
    Final_Data.append((bag_of_words(r),v))
#implementing naive bayes in unigram method
train_set, test_set = Final_Data[0:50], Final_Data[50:]

import nltk
import collections
from nltk.metrics.scores import (accuracy, precision, recall, f_measure) 
from nltk import metrics

refsets = collections. defaultdict(set)
testsets = collections.defaultdict(set)

classifier = nltk.NaiveBayesClassifier.train(train_set)
 
for i, (feats, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)

print("Naive Bayes Performance with Unigrams ")    
print("Accuracy:",nltk.classify.accuracy(classifier, test_set))
#calculating bullying recall
nb_classifier = nltk.NaiveBayesClassifier.train(train_set)

nbrefset = collections.defaultdict(set)
nbtestset = collections.defaultdict(set)
 
for i, (feats, label) in enumerate(test_set):
    nbrefset[label].add(i)
    observed = nb_classifier.classify(feats)
    nbtestset[observed].add(i)
print("UnigramNB Recall")
print('Bullying recall:', recall(nbtestset['Bullying'], nbrefset['Bullying']))
print("")
#implementing Decision tree classifier in unigram method
from nltk.classify import DecisionTreeClassifier

dt_classifier = DecisionTreeClassifier.train(train_set, 
                                             binary=True, 
                                             entropy_cutoff=0.8, 
                                             depth_cutoff=5, 
                                             support_cutoff=30)
refset = collections.defaultdict(set)
testset = collections.defaultdict(set)
 
for i, (feats, label) in enumerate(test_set):
    refset[label].add(i)
    observed = dt_classifier.classify(feats)
    testset[observed].add(i)
print("UnigramDT Recall")
print('Bullying recall:', recall(testset['Bullying'], refset['Bullying']))
print("")
#implementing Logistic regression in unigram method
from nltk.classify import MaxentClassifier

logit_classifier = MaxentClassifier.train(train_set, algorithm='gis', trace=0, max_iter=10, min_lldelta=0.5)

for i, (feats, label) in enumerate(test_set):
    refset[label].add(i)
    observed = logit_classifier.classify(feats)
    testset[observed].add(i)
print("UnigramsLogit Recall")
print('Bullying recall:', recall(testset['Bullying'], refset['Bullying']))
print("")
#bigram

from nltk import bigrams, trigrams
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

combined = zip(Tweet,Labels)
def bag_of_bigrams_words(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)  
    bigrams = bigram_finder.nbest(score_fn, n)  
    return bag_of_words(bigrams)
Final_Data2 =[]

for z, e in combined:
    bag_of_bigrams_words(z)
    Final_Data2.append((bag_of_bigrams_words(z),e))
#implement naive bayes in bigram method
import random
random.shuffle(Final_Data2)
print(len(Final_Data2))

train_set, test_set = Final_Data2[0:50], Final_Data2[50:]

import nltk
import collections
from nltk.metrics.scores import (accuracy, precision, recall, f_measure) 
from nltk import metrics



refsets = collections. defaultdict(set)
testsets = collections.defaultdict(set)

classifier = nltk.NaiveBayesClassifier.train(train_set)

 
for i, (feats, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)


print("Naive Bayes Performance with Unigrams ")    
print("Accuracy:",nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(n=10)
#implement Decision tree classifier in bigram method
from nltk.classify import DecisionTreeClassifier

dt_classifier = DecisionTreeClassifier.train(train_set, 
                                             binary=True, 
                                             entropy_cutoff=0.8, 
                                             depth_cutoff=5, 
                                             support_cutoff=30)
refset = collections.defaultdict(set)
testset = collections.defaultdict(set)
 
for i, (feats, label) in enumerate(test_set):
    refset[label].add(i)
    observed = dt_classifier.classify(feats)
    testset[observed].add(i)
print("BigramDT Recall")
print('Bullying recall:', recall(testset['Bullying'], refset['Bullying']))
print("")
#implement Logistic regression in bigram method
from nltk.classify import MaxentClassifier

logit_classifier = MaxentClassifier.train(train_set, algorithm='gis', trace=0, max_iter=10, min_lldelta=0.5)

for i, (feats, label) in enumerate(test_set):
    refset[label].add(i)
    observed = logit_classifier.classify(feats)
    testset[observed].add(i)
print("BigramsLogit Recall")
print('Bullying recall:', recall(testset['Bullying'], refset['Bullying']))
print("")

combined = zip(Tweet,Labels)

from nltk import bigrams, trigrams
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import TrigramAssocMeasures

def bag_of_trigrams_words(words, score_fn=TrigramAssocMeasures.chi_sq, n=200):
    trigram_finder = TrigramCollocationFinder.from_words(words)  
    trigrams = trigram_finder.nbest(score_fn, n)  
    return bag_of_words(trigrams)

#implementing naive bayes in trigram method
Final_Data3 =[]

for z, e in combined:
    bag_of_trigrams_words(z)
    Final_Data3.append((bag_of_trigrams_words(z),e))

import random
random.shuffle(Final_Data3)
print(len(Final_Data3))

train_set, test_set = Final_Data3[0:747], Final_Data3[747:]

import nltk
import collections
from nltk.metrics.scores import (accuracy, precision, recall, f_measure) 
from nltk import metrics


refsets = collections. defaultdict(set)
testsets = collections.defaultdict(set)

classifier = nltk.NaiveBayesClassifier.train(train_set)

 
for i, (feats, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)


print("Naive Bayes Performance with Trigrams ")    
print("Accuracy:",nltk.classify.accuracy(classifier, test_set))

print('bullying precision:', precision(refsets['Bullying'], testsets['Bullying']))
print('bullying recall:', recall(refsets['Bullying'], testsets['Bullying']))


classifier.show_most_informative_features(n=10)

#implementing decision tree classifier in trigram method

from nltk.classify import DecisionTreeClassifier

dt_classifier = DecisionTreeClassifier.train(train_set, 
                                             binary=True, 
                                             entropy_cutoff=0.8, 
                                             depth_cutoff=5, 
                                             support_cutoff=30)
refset = collections.defaultdict(set)
testset = collections.defaultdict(set)
 
for i, (feats, label) in enumerate(test_set):
    refset[label].add(i)
    observed = dt_classifier.classify(feats)
    testset[observed].add(i)
print("TrigramDT Recall")
print('Bullying recall:', recall(testset['Bullying'], refset['Bullying']))
print("")

#implementing Logistic regression in trigram method

from nltk.classify import MaxentClassifier

logit_classifier = MaxentClassifier.train(train_set, algorithm='gis', trace=0, max_iter=10, min_lldelta=0.5)

for i, (feats, label) in enumerate(test_set):
    refset[label].add(i)
    observed = logit_classifier.classify(feats)
    testset[observed].add(i)
print("TrigramsLogit Recall")
print('Bullying recall:', recall(testset['Bullying'], refset['Bullying']))
print("")
combined = zip(Tweet,Labels)

def bigrams_words(words, score_fn=BigramAssocMeasures.chi_sq,
n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bigrams

from nltk.collocations import TrigramCollocationFinder

# Import Bigram metrics - we will use these to identify the top 200 bigrams
from nltk.metrics import TrigramAssocMeasures

def trigrams_words(words, score_fn=TrigramAssocMeasures.chi_sq,
n=200):
    trigram_finder = TrigramCollocationFinder.from_words(words)
    trigrams = trigram_finder.nbest(score_fn, n)
    return trigrams


def bag_of_Ngrams_words(words):
    bigramBag = bigrams_words(words)
    
    #The following two for loops convert tuple into string
    for b in range(0,len(bigramBag)):
        bigramBag[b]=' '.join(bigramBag[b])
   
    trigramBag = trigrams_words(words)
    for t in range(0,len(trigramBag)):
        trigramBag[t]=' '.join(trigramBag[t])

    return bag_of_words(trigramBag + bigramBag + words)

Final_Data4 =[]

for z, e in combined:
    bag_of_Ngrams_words(z)
    Final_Data4.append((bag_of_Ngrams_words(z),e))
    
#implementing Naive bayes in n-gram method
import random
random.shuffle(Final_Data4)
print(len(Final_Data4))

train_set, test_set = Final_Data4[0:747], Final_Data4[747:]

import nltk
import collections
from nltk.metrics.scores import (accuracy, precision, recall, f_measure) 
from nltk import metrics


refsets = collections. defaultdict(set)
testsets = collections.defaultdict(set)

classifier = nltk.NaiveBayesClassifier.train(train_set)

 
for i, (feats, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)


print("Naive Bayes Performance with Ngrams ")    
print("Accuracy:",nltk.classify.accuracy(classifier, test_set))

classifier.show_most_informative_features(n=10)
print('bullying precision:', precision(refsets['Bullying'], testsets['Bullying']))
print('bullying recall:', recall(refsets['Bullying'], testsets['Bullying']))

#implementing decision tree classifier in n-gram method
from nltk.classify import DecisionTreeClassifier

dt_classifier = DecisionTreeClassifier.train(train_set, 
                                             binary=True, 
                                             entropy_cutoff=0.8, 
                                             depth_cutoff=5, 
                                             support_cutoff=30)
refset = collections.defaultdict(set)
testset = collections.defaultdict(set)
 
for i, (feats, label) in enumerate(test_set):
    refset[label].add(i)
    observed = dt_classifier.classify(feats)
    testset[observed].add(i)
print("NgramDT Recall")
print('Bullying recall:', recall(testset['Bullying'], refset['Bullying']))
print("")

#implementing Logistic regression in n-gram method
from nltk.classify import MaxentClassifier

logit_classifier = MaxentClassifier.train(train_set, algorithm='gis', trace=0, max_iter=10, min_lldelta=0.5)

for i, (feats, label) in enumerate(test_set):
    refset[label].add(i)
    observed = logit_classifier.classify(feats)
    testset[observed].add(i)
print("NgramsLogit Recall")
print('Bullying recall:', recall(testset['Bullying'], refset['Bullying']))
print("")

train_set, test_set = Final_Data[0:747], Final_Data[747:]

import nltk
import collections
from nltk.metrics.scores import (accuracy, precision, recall, f_measure)
nb_classifier = nltk.NaiveBayesClassifier.train(train_set)
nb_classifier.show_most_informative_features(10)

from nltk.classify.util import accuracy
print(accuracy(nb_classifier, test_set))

refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)
    
for i, (Final_Data, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = nb_classifier.classify(Final_Data)
    testsets[observed].add(i)
    
print('bullying precision:', precision(refsets['Bullying'], testsets['Bullying']))
print('bullying recall:', recall(refsets['Bullying'], testsets['Bullying']))
print('bullying F-measure:', f_measure(refsets['Bullying'], testsets['Bullying']))
print('not-bullying precision:', precision(refsets['Non-Bullying'], testsets['Non-Bullying']))
print('not-bullying recall:', recall(refsets['Non-Bullying'], testsets['Non-Bullying']))
print('not-bullying F-measure:', f_measure(refsets['Non-Bullying'], testsets['Non-Bullying']))

def bigrams_words(words, score_fn=BigramAssocMeasures.chi_sq,
n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bigrams

from nltk.collocations import TrigramCollocationFinder

# Import Bigram metrics - we will use these to identify the top 200 bigrams
from nltk.metrics import TrigramAssocMeasures

def trigrams_words(words, score_fn=TrigramAssocMeasures.chi_sq,
n=200):
    trigram_finder = TrigramCollocationFinder.from_words(words)
    trigrams = trigram_finder.nbest(score_fn, n)
    return trigrams


def bag_of_Ngrams_words(words):
    bigramBag = bigrams_words(words)
    
    #The following two for loops convert tuple into string
    for b in range(0,len(bigramBag)):
        bigramBag[b]=' '.join(bigramBag[b])
   
    trigramBag = trigrams_words(words)
    for t in range(0,len(trigramBag)):
        trigramBag[t]=' '.join(trigramBag[t])

    return bag_of_words(trigramBag + bigramBag + words)


# In[76]:


zl = zip(Tweet,Labels)

Final_Data = []

for k, v in zl:
    bag_of_words(k)
    Final_Data.append((bag_of_words(k),v))

import random
random.shuffle(Final_Data)

#splits the data around 70% of 500 *350 reviews* for both testing and training

train_set, test_set = Final_Data[0:778], Final_Data[778:]

#Now we will calculate accuracy, precision, recall, and f-measure using Naives Bayes classifier

import nltk
import collections
from nltk.metrics.scores import (accuracy, precision, recall, f_measure)
nb_classifier = nltk.NaiveBayesClassifier.train(train_set)
nb_classifier.show_most_informative_features(10)

from nltk.classify.util import accuracy
print(accuracy(nb_classifier, test_set))

refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)
    
for i, (Final_Data, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = nb_classifier.classify(Final_Data)
    testsets[observed].add(i)
    
print('bullying precision:', precision(refsets['Bullying'], testsets['Bullying']))
print('bullying recall:', recall(refsets['Bullying'], testsets['Bullying']))
print('bullying F-measure:', f_measure(refsets['Bullying'], testsets['Bullying']))
print('not-bullying precision:', precision(refsets['Non-Bullying'], testsets['Non-Bullying']))
print('not-bullying recall:', recall(refsets['Non-Bullying'], testsets['Non-Bullying']))
print('not-bullying F-measure:', f_measure(refsets['Non-Bullying'], testsets['Non-Bullying']))



