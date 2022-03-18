# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:19:31 2022

Topic : Handling Missing values or imputation
"""
'''
Problem Statement:  
Majority of the datasets have missing values, that might be because the data collected were not at regular intervals or the breakdown of instruments and so on. It is nearly impossible to build the proper model or in other words, get accurate results. The common techniques are either removing those records completely or substitute those missing values with the logical ones, there are various techniques to treat these types of problems.
1)	Prepare the dataset using various techniques to solve the problem, explore all the techniques available and use them to see which gives the best result.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('D:/DataSets/claimants.csv')
 
#FINDING OUTLIERS
plt.boxplot(data.CLMSEX)
plt.boxplot(data.CLMINSUR)
plt.boxplot(data.SEATBELT)
plt.boxplot(data.CLMAGE)

#check for null values
data.isna().sum()

#create an imputer object that fills 'Nan' values
#simple imputer 

from sklearn.impute import SimpleImputer

#imputing with mean strategy
si = SimpleImputer(missing_values=np.nan, strategy='mean') #mean is  taken becoz numeric data

data['CLMSEX'] = pd.DataFrame(si.fit_transform(data[['CLMSEX']]))

data['CLMSEX'].isna().sum()

si = SimpleImputer(missing_values=np.nan, strategy='mean') #mean is  taken becoz numeric data

data['CLMINSUR'] = pd.DataFrame(si.fit_transform(data[['CLMINSUR']]))

data['CLMINSUR'].isna().sum()

#Imputing median strategy

si = SimpleImputer(missing_values=np.nan, strategy='median') #mean is  taken becoz numeric data

data['SEATBELT'] = pd.DataFrame(si.fit_transform(data[['SEATBELT']]))

data['SEATBELT'].isna().sum()

si = SimpleImputer(missing_values=np.nan, strategy='median') #mean is  taken becoz numeric data

data['CLMAGE'] = pd.DataFrame(si.fit_transform(data[['CLMAGE']]))

data['CLMAGE'].isna().sum()

data.isna().sum()

# Conclusion : Missing values are handled using imputation methods like mean, median and mode 
