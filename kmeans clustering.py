# -*- coding: utf-8 -*-
"""
Name: BOTTA RAJU,
Batch ID: _05102021__________
Topic: K Means Clustering

"""
#1. East west airlines

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

air = pd.read_excel('D:/Data Set/EastWestAirlines.xlsx',1)
air.describe()
air.columns

air1 = air.drop('ID#', axis=1)

# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(air1.iloc[:, 0:])

###### scree	 plot or elbow curve ############
TWSS = []
k = list(range(1, 12))

from sklearn.cluster import	KMeans

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 2)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
air1['clust'] = mb # creating a  new column and assigning it to new column 

air1.head()
df_norm.head()

air1 = air1.iloc[:,0:]
air1.head()

air1.iloc[:, 0:].groupby(air1.clust).mean()

air1.to_csv("airlines.csv", encoding = "utf-8")

import os
os.getcwd()

============= ************ =================


#2. crime dataset

# 2.	Perform clustering for the crime data and identify the number of clusters            formed and draw inferences. Refer to crime_data.csv dataset.

# Business Problem : To identify the no.of clusters formed.

crime = pd.read_csv('D:/Data Set/crime_data.csv')
crime.describe()
crime.columns

#Data Preprocessing

# Normalization function 

def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(crime.iloc[:, 1:])

###### scree plot or elbow curve ############
TWSS = []
k = list(range(1, 5))

from sklearn.cluster import	KMeans

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 2)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
crime['clust'] = mb # creating a  new column and assigning it to new column 

crime.head()
df_norm.head()

crime = crime.iloc[:,0:]
crime.head()

crime.iloc[:, 0:].groupby(crime.clust).mean()

crime.to_csv("crime_data.csv", encoding = "utf-8")

import os
os.getcwd()

============= *********===========

# 3. Insurance dataset

#. Objective : To create clusters of persons falling in same type

insurance = pd.read_csv('D:\Data Set\Insurance Dataset.csv')
insurance.describe()
insurance.columns
insurance.isna().sum()

  # EDA
  #Univariate Analysis
  #Distribution plots

sns.FacetGrid(insurance, hue = 'Income', size = 5).map(sns.distplot,'Claims made')  
plt.hist(insurance['Claims made'], 10)  
  
  # CDF (Cumulatinve Distributive Function) & Probability Distribution Function
  
counts, bin_edges = np.histogram(insurance['Claims made'], bins=10, density = True)
plt.xlabel('Claims made')
pdf = counts/(sum(counts))
print("pdf=", pdf)
print("bin edges=", bin_edges)
cdf = np.cumsum(pdf)
print("cdf=", cdf)
plt.plot(bin_edges[0:], cdf) 
plt.plot(bin_edges[0:], pdf)

#Bivariate Analysis  
plt.bar(height=insurance.income, x = insurance['Claims made'])
plt.boxplot(insurance['Income'])  

#Data Preprocessing

# Normalization function 

def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(insurance.iloc[:, 0:])

###### scree plot or elbow curve ############
TWSS = []
k = list(range(1, 6))

from sklearn.cluster import	KMeans

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
insurance['clust'] = mb # creating a  new column and assigning it to new column 

insurance.head()
df_norm.head()

insurance = insurance.iloc[:,0:]
insurance.head()

insurance.iloc[:, 0:].groupby(insurance.clust).mean()

insurance.to_csv("Insurance Dataset.csv", encoding = "utf-8")

import os
os.getcwd()

========= *********** ==================

#4. telecom dataset
# Perform clustering analysis on the telecom dataset. The data is a mixture of both categorical and numerical data. It consists of the number of customers who churn. Derive insights and get possible information on factors that may affect the churn decision. Refer to Telco_customer_churn.xlsx dataset.

telco = pd.read_excel('D:\Data Set\Telco_customer_churn.xlsx')
telco.describe()
telco.columns

  # EDA
  #Univariate Analysis
  #Distribution plots
  
#  CDF (Cumulatinve Distributive Function) & Probability Distribution Function
  
#counts, bin_edges = np.histogram(telco['Total Revenue'], bins=10, density = True)
plt.hist(telco['Total Charges'], 10)  

plt.xlabel('Total Revenue')
pdf = counts/(sum(counts))
print("pdf=", pdf)
print("bin edges=", bin_edges)
cdf = np.cumsum(pdf)
print("cdf=", cdf)

plt.plot(bin_edges[1:], cdf) 
plt.plot(bin_edges[1:], pdf)

#Bivariate Analysis  
plt.bar(height = telco['Total Revenue'], x = telco['Total Charges'])
plt.boxplot(telco['Total Charges'])  
plt.scatter(telco['Total Revenue'], telco['Total Charges'])

#Data Preprocessing
#checking nan values
telco.isna().sum()
#dropping column
telco.drop('Customer ID', axis=1, inplace = True)

#Get dummies for categorical part
telco1 = pd.get_dummies(telco, drop_first=True) 

# Normalization function 

def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(telco1.iloc[:, 0:])

###### scree plot or elbow curve ############
TWSS = []
k = list(range(0, 29))

from sklearn.cluster import	KMeans

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)         
    
TWSS
 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
telco['clust'] = mb # creating a  new column and assigning it to new column 

telco.head()
df_norm.head()

telco = telco.iloc[:,0:]
telco.head()

telco.iloc[:, 0:].groupby(telco.clust).mean()

telco.to_csv("Telcom Dataset.csv", encoding = "utf-8")

import os
os.getcwd()

===================== ************** =========================

#5. auto insurance
auto = pd.read_csv('D:\Data Set\AutoInsurance.csv')
auto.describe()
auto.columns

#Data Preprocessing

# get dummies for categorical data
auto1 = pd.get_dummies(auto, drop_first=True) 

# Normalization function 
def norm_func(i):
    x = (i - i.min())/ (i.max() - i.min())
    return (x)
 
# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(auto1.iloc[:, 0:])

###### scree plot or elbow curve ############
TWSS = []
k = list(range(0, 29))

from sklearn.cluster import	KMeans

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)         
    
TWSS
 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
telco['clust'] = mb # creating a  new column and assigning it to new column 

telco.head()
df_norm.head()

telco = telco.iloc[:,0:]
telco.head()

telco.iloc[:, 0:].groupby(insurance.clust).mean()

telco.to_csv("Insurance Dataset.csv", encoding = "utf-8")

import os
os.getcwd()

