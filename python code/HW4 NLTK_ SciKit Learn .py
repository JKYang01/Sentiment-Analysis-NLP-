#!/usr/bin/env python
# coding: utf-8


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


# In[80]:


# modify the stopword list from english stop words 
# here I keep the negation words like "not","no","n't" because the negation can affect the sentiment
stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
             "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its","itself", 
             "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", 
             "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", 
             "an", "the", "and", "but","if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against",
             "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up",
             "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there",
             "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
             "only", "own", "same", "so", "than", "too", "very", "'s", "can", "will", "just", "should", "now","may","would","could"]
sentences = sentence_polarity.sents()
documents = [(sent,cat) for cat in sentence_polarity.categories() for sent in sentence_polarity.sents(categories = cat) ]
# shuffle the sort so that we could get rid of the effect of sort 
# and get the negtive and positive sentences in equal (or approch to equal)number 
random.shuffle(documents)


# # modify the top 2000 words of sentence_polarity corpas

# In[82]:


# Top 2000 most frequent words we use the tokenizer and filler the stop words 
# get all words 
all_words_list = [word for (sent,cat) in documents for word in sent]
# filter the word 
words_list = [word for word in all_words_list if word not in stopwords and word.isalpha() ]
words = nltk.FreqDist(words_list)
word_items = words.most_common(2000)
word_features = [word for (word, freq) in word_items]
#print(word_features[1:100])


# In[103]:


#The feature label will be ‘V_keyword’ for each keyword (aka word) in the word_features set
#and the value of the feature will be Boolean, according to whether the word is contained in that document.
def document_features(document, word_features):
    document_words = set(document) # got unique elements
    features = {}
    for word in word_features:
    # build featuresets that contains every feature of each sentences 
        if word in document_words:
            features['v_{}'.format(word)] = 1
        else :
            features['v_{}'.format(word)] = 0
    return features


featuresets = [(document_features(d,word_features),c) for (d,c) in documents]
#print(featuresets[0])


X = [list(d.values()) for (d,c) in featuresets]
#print(X[0])


# separate the data
# X =  list(map(lambda x : list(x[0].values()), featuresets))

y = [c for (s,c) in featuresets]

train_X,test_X,train_y,test_y = train_test_split(X,y,test_size = 0.1,random_state = 0)

print(train_X[0])
#print(train_y[0])


# In[111]:


model_1 = MultinomialNB()
model_1.fit(train_X, train_y)
pred_y = model_1.predict(test_X)


# In[139]:


# print the accuracy 
print("The Accuracy Score:")
print(model_1.score(test_X,test_y))
print("\n")

# print confusion matrix
print('The Confusion Matrix:\n')
cm = confusion_matrix(test_y,pred_y,labels=['pos','neg'])
print(cm)
print("\n")

# print crossvalidation score 
print("The avarage of Crossvalidation Scores:")
scores = cross_val_score(model_1, X, y, cv=3)
avg=sum(scores)/len(scores)
print(avg)
print("\n")

# print the classification report
print("The Classification Report:")
print(classification_report(test_y, pred_y))


# # Adding Features from a Sentiment Lexicon

# In[129]:


# read the subjective_lexicon file
# each line contains 5 catagories. each of the categories and their values is joined with '='
SLpath = r'C:\Users\wangtao\Desktop\python_work\NLP_stuff\subjclueslen1-HLTEMNLP05.tff'

with open (SLpath,'r') as flexicon:
    # build the dictionary that contains the words and its 
    subject_lex = {}

    for line in flexicon:
        
        f = line.split()
        strength = f[0].split('=')[1]
        word = f[2].split('=')[1]
        tag = f[3].split('=')[1]
        stemmed = f[4].split('=')[1]
        polarity = f[5].split('=')[1]
        
        if (stemmed == 'y'):
            isstemmed = True
        else:
            isstemmed = False 
            
        a = [strength, tag, isstemmed, polarity]
        
        subject_lex[word]=a
         
        


# In[130]:


def SL_features(document, word_features, subject_lex):
    document_words = set(document)
    features = {}
    for word in word_features:
        if word in document_words:
            features['v_{}'.format(word)] = 1 # the dictionary {"v_word":1}
        else:
            features['v_{}'.format(word)] = 0 # the dictionary{"v_word":0}
# count variables for the 4 classes of subjectivity
    weakPos = 0
    strongPos = 0
    weakNeg = 0
    strongNeg = 0
    for word in document_words:
        if word in subject_lex:
            strength, posTag, isstemmed, polarity = subject_lex[word]
            if strength == 'weaksubj' and polarity == 'positive':
                weakPos += 1
            if strength == 'strongsubj' and polarity == 'positive':
                strongPos += 1
            if strength == 'weaksubj' and polarity == 'negative':
                weakNeg += 1
            if strength == 'strongsubj' and polarity == 'negative':
                strongNeg += 1
            features['positivecount'] = weakPos + (2 * strongPos)
            features['negativecount'] = weakNeg + (2 * strongNeg)
        else:
            features['positivecount'] = 0
            features['negativecount'] = 0
    return features


# In[131]:


SL_featuresets = [(SL_features(d, word_features,subject_lex), c) for (d,c) in documents]
#print(SL_featuresets[0])


# In[133]:


X_1 = [list(f.values()) for (f,c) in SL_featuresets]
y_1 = [c for (f,c) in SL_featuresets]

train_X_1,test_X_1,train_y_1,test_y_1 = train_test_split(X_1,y_1,test_size = 0.1,random_state = 0)


# In[135]:


model_2 = MultinomialNB()
model_2.fit(train_X_1, train_y_1)
pred_y_1 = model_2.predict(test_X_1)


# In[138]:


print("The Accuracy Score:")
print(model_2.score(test_X_1,test_y_1))
print("\n")

# print confusion matrix
print('The Confusion Matrix:')
cm = confusion_matrix(test_y_1,pred_y_1,labels=['pos','neg'])
print(cm)
print("\n")

# print crossvalidation score 
print("The avarage of Crossvalidation Scores:")
scores = cross_val_score(model_2, X_1, y_1, cv=3)
avg=sum(scores)/len(scores)
print(avg)
print("\n")

# print the classification report
print("The Classification Report:")
print(classification_report(test_y_1, pred_y_1))


# In[140]:


negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather',
'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor']


# In[168]:


def NOT_features(document, word_features, negationwords):
    features = {}
    for word in word_features:
        features['v_{}'.format(word)] = 0
        features['v_NOT{}'.format(word)] = 0
# go through document words in order
    for i in range(0, len(document)):
        word = document[i]
        if ((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
            i += 1
            if document[i] in word_features:
                features['v_NOT{}'.format(document[i])] = 1
        else:
            if document[i] in word_features:
                features['v_{}'.format(word)] = 1
    return features


# In[169]:


NOT_featuresets = [(NOT_features(d, word_features, negationwords),c) for (d,c)in documents]

#NOT_featuresets[0][0]['v_NOTlike']


# In[170]:

X_2 = [list(f.values()) for (f,c) in NOT_featuresets]

y_2 = [c for (f,c) in NOT_featuresets]

train_X_2,test_X_2,train_y_2,test_y_2 = train_test_split(X_2,y_2,test_size = 0.1,random_state = 0)


model_3 = MultinomialNB()
model_3.fit(train_X_2, train_y_2)
pred_y_2 = model_3.predict(test_X_2)


# In[172]:


print("The Accuracy Score:")
print(model_3.score(test_X_2,test_y_2))
print("\n")

# print confusion matrix
print('The Confusion Matrix:')
cm = confusion_matrix(test_y_2,pred_y_2,labels=['pos','neg'])
print(cm)
print("\n")

# print crossvalidation score 
print("The avarage of Crossvalidation Scores:")
scores = cross_val_score(model_3, X_2, y_2, cv=3)
avg=sum(scores)/len(scores)
print(avg)
print("\n")

# print the classification report
print("The Classification Report:")
print(classification_report(test_y_2, pred_y_2))


# In[197]:


# Do the prediction 
rev_data = pd.read_csv('revtxt.csv')
rev_data.head()


# In[201]:


sents_list = rev_data['review'].tolist()
#print(sents_list[0:10])


# In[203]:


NOT_featuresets_test = [NOT_features(d.split(), word_features,negationwords) for d in sents_list[0:1000]]

# In[211]:

pos_pred = []
neg_pred = []
X = list(map(lambda x: list(x.values()), NOT_featuresets_test))
print(len(sents_list[0:1000]))
for i in range(len(sents_list[0:1000])):

    if model_3.predict([X[i]]) == 'pos':
        pos_pred.append(sents_list[i]+","+"pos"+"\n")

    else:
        neg_pred.append(sents_list[i]+","+"neg"+"\n")

#print(pos_pred)
print(neg_pred)
#with open('pos5.txt','w') as f:
#    f.writelines(pos_pred)

#with open('neg5.txt','w') as f:
#    f.writelines(neg_pred)


# In[212]:


with open('positive_review.txt','w') as f:
    f.writelines(pos_pred)

with open('negative_review.txt','w') as f:
    f.writelines(neg_pred)


# In[ ]:


rev_data["sentiment prediction"] = rev_data["review"].apply(lambda x : prediction(x))

