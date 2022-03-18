# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 20:38:09 2022

Name: BOTTA RAJU_____________ Batch ID: _05102021__________
Topic: K-Nearest Neighbors 

"""

import numpy as np
import pandas as pd

glass = pd.read_csv('D:\datasets\glass.csv')

#Normalization Function

def norm_func(i):
    x = (i-i.min()/i.max()-i.min())
    return (x)

#Normalizing data frame

glass_n = norm_func(glass.iloc[:,:])
glass_n.describe()

x = np.array(glass_n.iloc[:,0:9])  #Predictor
y = np.array(glass['Type'])  #Target

from sklearn.model_selection import train_test_split #Model selection

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25)
 
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=15) # knn Model 
knn.fit(x_train, y_train) 

pred = knn.predict(x_test) #predict model
pred

#Evaluete the model

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, pred)) #accuracy on train data
pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predictions'])

#Error on train data
pred_train = knn.predict(x_train)
print(accuracy_score(y_train, pred_train)) # accuracy on test data
pd.crosstab(y_train,pred_train,rownames=['Actual'], colnames=['Predictoins'])

acc = []

#  Running KNN Algorithm for 3-50 nearest neighbors (odd numbers)

# sorting accuracy values

for i in range(3,50,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(x_train, y_train)
    train_acc = np.mean(neigh.predict(x_train)==y_train)
    test_acc = np.mean(neigh.predict(x_test)==y_test)
    acc.append([train_acc, test_acc]) 

import matplotlib.pyplot as plt

#train accuracy plot
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")

# =================== ************************* ========================== 

# 2.	A National Zoopark in India is dealing with the problem of segregation of the animals based on the different attributes they have. Build a KNN model to automatically classify the animals. Explain any inferences you draw in the documentation.

zoo = pd.read_csv('D:\Data Set\Zoo.csv')
   
# Normalized data frame (considering the numerical part of data)

#zoo_n = norm_func(zoo.iloc[:,1:-1])
#zoo_n.describe()

x = np.array(zoo.iloc[:,1:17]) #predictors
y = np.array(zoo['type']) # target
 
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train, y_train)

pred = knn.predict(x_test)
pred

   # Evalute the model
   
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, pred)) 
pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predictions'])

            #Error on train data
            
pred_train = knn.predict(x_train)
print(accuracy_score(y_train, pred_train))
pd.crosstab(y_train, pred_train, rownames=['Actual'], colnames=['Predictions'])

acc = []

#  Running KNN Algorithm for 3-50 nearest neighbors (odd numbers)

# sorting accuracy values

for i in range(3,50,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(x_train, y_train)
    
    train_acc = np.mean(neigh.predict(x_train)==y_train)
    test_acc = np.mean(neigh.predict(x_test)==y_test)   

    acc.append([train_acc, test_acc]) 

import matplotlib.pyplot as plt

#train accuracy plot
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")


