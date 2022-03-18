# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:22:35 2022

Topic : Feature Scaling - Normalization & Standardization

"""

'''
Problem Statement: 
Data is one of the most important assets. It is often common that data is stored in distinct systems with different formats and scales. These seemingly small differences in how the data is stored can result in misinterpretations and inconsistencies in your analytics. Inconsistency can make it impossible to deliver reliable information to management for good decision making. We have the preprocessing techniques to make the data uniform. Explore the various techniques to have reliable uniform standard data, 


1)	Prepare the dataset by performing the preprocessing techniques, to have the standard scale to data
'''

import pandas as pd
import numpy as np

d = pd.read_csv('D:/DataSets/Seeds_data.csv')

from sklearn.preprocessing import StandardScaler
#initialise the standard scalar

scalar = StandardScaler()
#To scale the data
df = scalar.fit_transform(d)

#convert the array back to dataframe
dataset = pd.DataFrame(df)
dataset.describe()


#Normalisation

data = pd.get_dummies(d, drop_first=True)

#for normalisation custom function

def norm_func(i):
    x = (i-i.min()/(i.max()-i.min()))
    return (x)
df_norm =norm_func(d)
df_norm.describe()

# Normalisation Using sklearn module

# spliting training and testing data
from sklearn.model_selection import train_test_split

X = d.iloc[:,1:8]
y = d['Type']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=27)

from sklearn.preprocessing import MinMaxScaler
# fit scaler on training data
norm = MinMaxScaler().fit(X_train)
# transform training data
X_train_norm = norm.transform(X_train)

# transform testing data
X_test_norm = norm.transform(X_test)


    