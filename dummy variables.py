# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:13:48 2022

@author: LENOVO

Topic : Dummy Variable creation
"""
'''
    Problem Statement: 
Data is one of the most important assets. It is often common that data is stored in distinct systems with different formats and forms. Non-numeric form of data makes it tricky while developing mathematical equations for prediction models. We have the preprocessing techniques to make the data convert to numeric form. Explore the various techniques to have reliable uniform standard data

1)	Prepare the dataset by performing the preprocessing techniques, to have the all the features in numeric format.
'''

import numpy as np
import pandas as pd

df = pd.read_csv('D:/DataSets/animal_category.csv')
df.columns
df.shape

#drop nominal columns 

df.drop(['Index'], axis=1, inplace=True)
df.dtypes

#create dummy variables

df_dum = pd.get_dummies(df)
df_dum1 = pd.get_dummies(df, drop_first=True)

#one-hot encoding

df.columns
df = df[['Animals','Gender','Homly','Types']]

from sklearn.preprocessing import OneHotEncoder
#CREATING instance for onehotencoder

ohe = OneHotEncoder()
ohe_df = pd.DataFrame(ohe.fit_transform(df.iloc[:,2:]).toarray())

#categorical columns are onehot encoded

#Label Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
#split data into input & output variables

x = df.iloc[:,0:3]
y = df['Types']

df.columns

#label encoding input variables
x['Animals'] = le.fit_transform(x['Animals'])
x['Gender'] = le.fit_transform(x['Gender'])
x['Homly'] = le.fit_transform(x['Homly'])

#label encoding output variables
y = le.fit_transform(y)
y = pd.DataFrame(y)

#concatenate x & y

df_new = pd.concat([x,y], axis =1)

df_new = df_new.rename(columns={0:'Type'})
df_new


Conclusion: Created dummies for all categorical variables, categorical columns are onehot encoded and categorical columns are label encoded.

