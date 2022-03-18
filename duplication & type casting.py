# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:17:39 2022

Topic: Duplication & Type casting

"""
'''
Problem statement: 
Data collected may have duplicate entries, that might be because the data collected were not at regular intervals or any other reason. To build a proper solution on such data will be a tough ask. The common techniques are either removing duplicates completely or substitute those values with a logical data. There are various techniques to treat these types of problems.

Q1. For the given dataset perform the type casting (convert the datatypes, ex. float to int)
Q2. Check for the duplicate values, and handle the duplicate values (ex. drop)
Q3. Do the data analysis (EDA)?
Such as histogram, boxplot, scatterplot etc

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('D:/DataSets/OnlineRetail.csv')

#1 Type Casting
data_quantity = data.Quantity.astype('float32')
data_unitprice = data.UnitPrice.astype('int32')

#2 Checking duplicate values
dup1 = data.duplicated()
dup1
sum(dup1)
data1 = data.drop_duplicates()
data1

data_quantity = data.Quantity.astype('float32')
data_unitprice = data.UnitPrice.astype('int32')

dup1 = data.duplicated()
dup1
sum(dup1)
data1 = data.drop_duplicates()
data1

#3      Exploratory Data Analysis

#Measures of Central Tendency 

data.Quantity.mean()
data.Quantity.median()
data.Quantity.mode()

#Measures of Dispersion
data.Quantity.var()
data.Quantity.std()
range = max(data.Quantity) - min(data.Quantity)
range

#Skewness
data.Quantity.skew() # negatively skewed

#Kurtosis
data.Quantity.kurt() # leptokurtic

#Measures of Central Tendency 

data.UnitPrice.mean()
data.UnitPrice.median()
data.UnitPrice.mode()

#Measures of Dispersion
data.UnitPrice.var()
data.UnitPrice.std()
range = max(data.UnitPrice) - min(data.UnitPrice)
range

#skewness
data.UnitPrice.skew()  #postively skewed or right skewed

#kurtosis
data.UnitPrice.kurt() #leptokurtic 


#Visualisation

data.shape
plt.bar(height=data.UnitPrice, x='Quantity')

#histogram
plt.hist(data.UnitPrice)  #To know data is normally distributed or not

#box plot
plt.boxplot(data.UnitPrice) #to know outliers are there or not
#the distribution is not normal

plt.boxplot(data.Quantity)
#the distribution is not normal

#scatter plot
plt.scatter(y=data.UnitPrice, x=data.Quantity, alpha=0.5 )
plt.show()


# Conclusion: Verified duplication and Typecasting (converting one datatype to another) has done.
