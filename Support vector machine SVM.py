# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 14:48:12 2022

Name: _BOTTA RAJU____________ Batch ID: _05102021__________
Topic: SVM

"""

# Objective: 1) to get insights on how densely the area is populated and the income levels of residents. Using the Support Vector Machines algorithm on the given dataset.

import pandas as pd
import numpy as np

construct_train = pd.read_csv(r"D:/Data Set/construction_firm.csv")
construct_train.columns
construct_test = pd.read_csv('D:/Data Set/construction_test.csv')
construct_test

# null values checking

construct_train.isnull().sum()

# label encoding test data

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
construct_test['workclass'] = le.fit_transform(construct_test['workclass']) 
construct_test['education'] = le.fit_transform(construct_test['education']) 
construct_test['maritalstatus'] = le.fit_transform(construct_test['maritalstatus']) 
construct_test['occupation'] = le.fit_transform(construct_test['occupation']) 
construct_test['relationship'] = le.fit_transform(construct_test['relationship']) 
construct_test['race'] = le.fit_transform(construct_test['race']) 
construct_test['sex'] = le.fit_transform(construct_test['sex']) 
construct_test['native'] = le.fit_transform(construct_test['native']) 
construct_test['Salary'] = le.fit_transform(construct_test['Salary']) 

# checking null values for test data

construct_test.isnull().sum()

from sklearn.svm import SVC 

from sklearn.model_selection import train_test_split

train, test = train_test_split(construct_train, construct_test, test_size=0.20)

train_x = construct_train.iloc[:,0:13]
train_y = construct_train['Salary']

test_x = construct_test.iloc[:,0:13]
test_y = construct_test['Salary']

# Kernel = linear

svc_linear = SVC(kernel='linear')
svc_linear.fit(train_x, train_y)

# predict

svc_linear_pred = svc_linear.predict(test_x)

np.mean(svc_linear_pred==test_y)

# Kernel = rbf

svc_rbf = SVC(kernel='rbf')
svc_rbf.fit(train_x, train_y)

# predict

svc_rbf_pred = svc_rbf.predict(test_x)

np.mean(svc_rbf_pred==test_y)

# ============================ ************************** ================

# 2. Objective: to predict the size of the burnt area in forest fires annually so that they can be better prepared in future calamities. 

forest_fires = pd.read_csv(r'D:\Data Set\forestfires.csv')


forest_fires.columns

forest_fires.isnull().sum()

# label encoding test data

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

forest_fires['month'] = le.fit_transform(forest_fires['month']) 
forest_fires['day'] = le.fit_transform(forest_fires['day']) 
forest_fires['size_category'] = le.fit_transform(forest_fires['size_category']) 

from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split

train, test = train_test_split(forest_fires, test_size=0.20)

train_x = forest_fires.iloc[:,0:30]
train_y = forest_fires['size_category']

test_x = construct_test.iloc[:,0:30]
test_y = construct_test['size_category']

# Kernel = linear

forest_linear = SVC(kernel='linear')
forest_linear.fit(train_x, train_y)

# predict

forest_linear_pred = svc_linear.predict(test_x)

np.mean(forest_linear_pred==test_y)

# Kernel = rbf

forest_rbf = SVC(kernel='rbf')
forest_rbf.fit(train_x, train_y)

# predict
forest_rbf_pred = svc_rbf.predict(test_x)

np.mean(forest_rbf_pred==test_y)

# =============================== ******************** =======================
