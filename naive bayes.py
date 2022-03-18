# -*- coding: utf-8 -*-
"""
Name: _BOTTA RAJU____________ Batch ID: 05102021___________
Topic:  Naïve Bayes

"""

import numpy as np # linear algebra
import pandas as pd # data processing, 

# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

# train a Gaussian Naive Bayes classifier on the training set
from sklearn.naive_bayes import GaussianNB

#Loading the data set
train = pd.read_csv('D:\Data Set\SalaryData_Train.csv')
test = pd.read_csv('D:\Data Set\SalaryData_Test.csv')
train1 = train.copy()
test1 = test.copy()
#checking for null values
train.isnull().sum()
test.isnull().sum()

x_train = train1.drop(['Salary'], axis = 1)
y_train = train1['Salary']

x_test = test1.drop(['Salary'], axis = 1)
y_test = test1['Salary']

# import category encoders
# pip install category_encoders
import category_encoders as ce
# encode remaining variables with one-hot encoding
x_train.columns
encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex', 'native'])

x_train = encoder.fit_transform(x_train)

x_test = encoder.fit_transform(x_test)

x_train.head()
x_train.shape
x_test.head()
x_test.shape

#scaling
cols = x_train.columns
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.fit_transform(x_test)
x_train = pd.DataFrame(x_train, columns=[cols])
x_test = pd.DataFrame(x_test, columns=[cols])
x_train.head()
x_test.head()

# instantiate the model
gnb = GaussianNB()

# fit the model
gnb.fit(x_train, y_train)

#prediction for test data
y_pred = gnb.predict(x_test)

#Accuracy score for test data
accuracy_score(y_test, y_pred)

#prediction for train data
y_pred = gnb.predict(x_train)

#Accuracy score for train data
accuracy_score(y_train, y_pred)	

# ===================== *************************** *************************
	 


# 2. Car Ad

car_ad = pd.read_csv(r'D:\Data Set\NB_Car_Ad.csv')
car_ad.drop(['User ID'], axis = 1 ,inplace = True)
car_ad.columns

car_ad['Gender'] = car_ad['Gender'].map({'Male': 1, 'Female': 0}).astype(int)

x = car_ad.drop(['Purchased'], axis = 1)
y = car_ad['Purchased']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

x_train.head()

cols = x_train.columns
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

x_train = pd.DataFrame(x_train, columns=[cols])
x_test = pd.DataFrame(x_test, columns=[cols])
x_train.head()
x_test.head()

# instantiate the Guassian Naive Bayes model
gnb = GaussianNB()

# fit the model
gnb.fit(x_train, y_train)

#prediction for test data
y_pred = gnb.predict(x_test)

#Accuracy score for test data
accuracy_score(y_test, y_pred)

#prediction for train data
y_pred = gnb.predict(x_train)

#Accuracy score for train data
accuracy_score(y_train, y_pred)  # 90% 

 #======================**********************============                      -------------------------- 

# Using Bernoulli Niave Bayes model

# instatiate the Bernoulli Naive Bayes model

from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
#fitting model
bnb.fit(x_train, y_train)

#predicting model
y_pred = bnb.predict(x_test)

#accuracy of model
accuracy_score(y_test, y_pred) # 80% 

# Guaissian Naive Bayes model (90%) gives more accuracy than Bernoulli Naive Bayes model..

# ======================== ******************* ===================

# 3. disaster tweets 
 
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

# Loading the data set
tweet = pd.read_csv(r"D:\Data Set\Disaster_tweets_NB.csv")

tweet.drop(['location','id'], axis = 1, inplace = True)
tweet.columns

tweet.fillna("Unknown", inplace = True)
tweet.isnull().sum()

# cleaning data 
import re # regular expression 
stop_words = []
# Load Stopwords
  
from nltk.corpus import stopwords

stop_words = str(stopwords.words('English')) # 179 pre defined stop words
print(stop_words)
    
stop_words = stop_words.split("\n")
   
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))

tweet.text = tweet.text.apply(cleaning_text)
type(tweet.text)
type(tweet)

# removing empty rows
tweet = tweet.loc[tweet.text != " ",:]

# CountVectorizer
# Convert a collection of text documents to a matrix of token counts
# splitting data into train and test data sets 

tweet_train, tweet_test = train_test_split(tweet, test_size = 0.2)

# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]

# Defining the preparation of email texts into word count matrix format - Bag of Words
tweet_bow = CountVectorizer(analyzer = split_into_words).fit(tweet.text)

# Defining BOW for all messages
tweet_matrix = tweet_bow.transform(tweet.text)

# For training messages
train_tweet_matrix = tweet_bow.transform(tweet_train.text)

# For testing messages
test_tweet_matrix = tweet_bow.transform(tweet_test.text)

# Learning Term weighting and normalizing on entire emails
tfidf_transformer = TfidfTransformer().fit(tweet_matrix)

# Preparing TFIDF for train emails
train_tfidf = tfidf_transformer.transform(train_tweet_matrix)
train_tfidf.shape # (row, column)

# Preparing TFIDF for test emails
test_tfidf = tfidf_transformer.transform(test_tweet_matrix)
test_tfidf.shape #  (row, column)

# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB

tweet.columns
# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf, tweet_train.target)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == tweet_test.target)
accuracy_test_m

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, tweet_test.target) 

pd.crosstab(test_pred_m, tweet_test.target)

# Training Data accuracy
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m == tweet_train.target)
accuracy_train_m

