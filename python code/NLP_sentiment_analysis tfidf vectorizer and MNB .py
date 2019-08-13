#!/usr/bin/env python
# coding: utf-8

# # using sklearn tfidf vectorizer and MNB to do the classification

# In[3]:


import nltk
import random
from nltk.corpus import stopwords
from nltk.corpus import sentence_polarity
from nltk.tokenize import word_tokenize
import pandas as pd

from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score


# In[4]:


sentences = sentence_polarity.sents()
documents = [(sent,cat) for cat in sentence_polarity.categories() for sent in sentence_polarity.sents(categories = cat) ]
# shuffle the sort so that we could get rid of the effect of sort 
# and get the negtive and positive sentences in equal (or approch to equal)number 
random.shuffle(documents)


# In[5]:


stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
             "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its","itself", 
             "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", 
             "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", 
             "an", "the", "and", "but","if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against",
             "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up",
             "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there",
             "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
             "only", "own", "same", "so", "than", "too", "very", "'s", "can", "will", "just", "should", "now","may","would","could"]


# In[6]:


# build a dataframe 
X = [' '.join(d) for (d,c) in documents]
y = [c for (d,c) in documents]

data = {"tag":y,"sents":X}
df = pd.DataFrame(data)

sents = df["sents"].values
tags  = df["tag"].values

# separate the data
train_X,test_X,train_y,test_y = train_test_split(sents,tags,test_size = 0.1)

print(train_X[0])
#print(train_y[0])


# In[7]:


def tokenize(text):
    # in the first time I found there are some non-word symbles #!? come up 
    # so I add in isalpha() in the tokenizing step to make sure it is a word
    token = [e for e in word_tokenize(text) if e.isalpha()]
    return token


# In[8]:


#Use nltk_tokenizer to tokenize the words and clean the stop words which might be the noise
#here we do not stemm the words because some sentiment words which are adjective might be the derivation form
tfidf = TfidfVectorizer(tokenizer = tokenize, stop_words = stopwords)


# In[9]:


train_X_vec = tfidf.fit_transform(train_X)
test_X_vec = tfidf.transform(test_X)
#print(train_X_vec)


# In[10]:


model4 = MultinomialNB()
model4.fit(train_X_vec, train_y)
pred_y = model4.predict(test_X_vec)


# In[31]:


from sklearn.pipeline import Pipeline

print("The Accuracy Score:")
print(model4.score(test_X_vec,test_y))
print("\n")

# print confusion matrix
print('The Confusion Matrix:')
cm = confusion_matrix(test_y,pred_y,labels=['pos','neg'])
print(cm)
print("\n")

# print crossvalidation score 
print("The avarage of Crossvalidation Scores:")
nb_clf_pipe = Pipeline([('vect', TfidfVectorizer(encoding='latin-1', binary=True, 
                                                 tokenizer = tokenize)),('nb', MultinomialNB())])
scores = cross_val_score(nb_clf_pipe, X, y, cv=10)
avg=sum(scores)/len(scores)
print(avg)

print("\n")

# print the classification report
print("The Classification Report:")
print(classification_report(test_y, pred_y))


# In[11]:


rev_data = pd.read_csv("revtxt.csv")


# In[12]:


def prediction (text):
    text = [text]
    text_vect = tfidf.transform(text)
    pred = model4.predict(text_vect)
    return pred


# In[13]:


rev_data["sentiment prediction"] = rev_data["review"].apply(lambda x : prediction(x))


# In[15]:


rev_data.head()


# In[16]:


rev_data.to_csv('predic_rev.csv')


# In[33]:


df2= pd.read_csv('predic_rev.csv')
df2[df2.apply(lambda x: x['sentiment prediction']=="['pos']", axis = 1)]


# In[34]:


df2[df2.apply(lambda x: x['sentiment prediction']=="['neg']", axis = 1)]


# In[ ]:




