# -*- coding: utf-8 -*-
"""
Hierarchical Clustering analysis
 Batch id - 05102021
@author: BOTTA RAJU..
"""

# EastWest Airlines

  # importing required modules

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#loading the dataset

dataset = pd.read_excel('/content/EastWestAirlines.xlsx',1)
dataset.head()
dataset.columns
dataset.describe
dataset.info

dataset.columns

# Data Pre -Processing

   #  Data Cleaning 
data = dataset.drop('ID#', axis=1)

#Verifying null values
data.isna().sum()

plt.scatter(data['Balance'], data['Days_since_enroll'])

plt.hist(data['Flight_miles_12mo'], 10)  

plt.boxplot(data['Balance'])  


#Bivariate Analysis  
plt.bar(height=data.Balance, x = data['Award?'])
  
# EDA

  #Univariate Analysis
  #Distribution plots
  
sns.FacetGrid(data, hue = 'Balance', size = 5).map(sns.distplot,'Award?')  
  
  # CDF (Cumulatinve Distributive Function) & Probability Distribution Function
  
counts, bin_edges = np.histogram(data['Award?'], bins=10, density = True)
plt.xlabel('Award')
pdf = counts/(sum(counts))
print("pdf=", pdf)
print("bin edges=", bin_edges)
cdf = np.cumsum(pdf)
print("cdf=", cdf)
plt.plot(bin_edges[1:], cdf) 
plt.plot(bin_edges[1:], pdf)


  # Normalization function 
def norm_func(i):
    x = (i-i.min())/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(data.iloc[:, 0:])
df_norm.describe()


# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));
plt.title('Hierarchical Clustering Dendrogram');
plt.xlabel('Balance');
plt.ylabel('Award')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 5, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

data['clust'] = cluster_labels # creating a new column and assigning it to new column 

data = data.iloc[:, [11,0,1,2,3,4,5,6,7,8,9,10]]
data.head()

# Aggregate mean of each cluster
data.iloc[:, 2:].groupby(data.clust).mean()

# creating a csv file 
data.to_csv("EastWestAirlines.xlsx", encoding = "utf-8")

import os
os.getcwd()

# ===================== ***************** ========================

#2. crime data set

crime = pd.read_csv("/content/crime_data.csv")

crime.describe()
crime.info()
crime.columns

crime.rename(columns = {'Unnamed':'state'}, inplace= True)
crime

crime.columns

plt.hist(crime['Murder'], 10)  

#Bivariate Analysis  
plt.bar(height = crime['Murder'], x = crime['Rape'])

plt.boxplot(crime['Rape'])  

# Normalization function 
def norm_func(i):
    x = (i-i.min())/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(crime.iloc[:, 1:4])
df_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df_norm, method = "complete", metric = "euclidean")

# Dendrogram

plt.figure(figsize=(15, 8));
plt.title('Hierarchical Clustering Dendrogram');
plt.xlabel('Index');plt.ylabel('Unanamed')
sch.dendrogram(z, leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
    )
plt.show()

# 5 clusters are formed from the above dendrogram

# Now applying AgglomerativeClustering choosing 2 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 2, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)
crime['cluster'] = cluster_labels  # creating a new column and assigning it to new column 

crime = crime.iloc[:,0:4]
crime.head()

# Aggregate mean of each cluster
crime.iloc[:,0:4].groupby(crime['cluster']).mean()

# creating a csv file 
crime.to_csv('crime_data.csv', encoding = 'utf-8')

import os
os.getcwd()

# ========= ************** ================

# 3.Telicom

telco = pd.read_excel('/content/Telco_customer_churn.xlsx')
telco.describe()
telco.info()
telco.columns
telco.drop(['Customer ID'], axis=1, inplace =True)

telco.columns

   # EDA
  #Univariate Analysis
  #Distribution plots
  
plt.hist(telco['Total Charges'], 10)  
plt.bar(height = telco['Total Revenue'], x = telco['Total Charges'])
plt.boxplot(telco['Total Charges'])  
plt.scatter(telco['Total Revenue'], telco['Total Charges'])

# get dummies for categorical data

telco1 = pd.get_dummies(telco, drop_first=True) 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(telco1.iloc[:,1:])
df_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));
plt.title('Hierarchical Clustering Dendrogram');
plt.xlabel('Index');plt.ylabel('Total Revenue')
sch.dendrogram(z, leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
    )
plt.show()


# 3 clusters are formed from the above dendrogram

# Now applying AgglomerativeClustering choosing 
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)
telco1clust = cluster_labels  # creating a new column and assigning it to new column 

telco1 = telco1.iloc[:,0:36]
telco1.head()

 # Aggregate mean of each cluster
telco1.iloc[:,0:4].groupby(telco1clust).mean()

# creating a csv file 
telco1.to_csv('crime_data.csv', encoding = 'utf-8')

import os
os.getcwd()

# ================= ************ ================

# auto insurance

# importing required modules

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

insur = pd.read_csv('/content/AutoInsurance.csv')

insur.info()
insur.describe()  
insur.columns
insur.drop(['Customer','State'], axis=1, inplace =True)

   # EDA
  #Univariate Analysis
  #Distribution plots
  
plt.hist(insur['Total Claim Amount'], 5)  

#Bivariate Analysis  
plt.bar(height = insur['Income'], x = insur['Total Claim Amount'])

plt.boxplot(insur['Total Claim Amount'])

plt.scatter(insur['Total Claim Amount'], insur['Income'])

# Data Pre processing
# Normalization 

# get dummies for categorical data

insur1 = pd.get_dummies(insur, drop_first=True) 
#Normalization
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(insur1.iloc[:,0:])
df_norm.describe()


# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));
plt.title('Hierarchical Clustering Dendrogram');
plt.xlabel('Income');plt.ylabel('Total Claim Amount')
sch.dendrogram(z, leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
    )
plt.show()

# 3 clusters are formed from the above dendrogram

# Now applying AgglomerativeClustering choosing 
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)
insur['clust'] = cluster_labels  # creating a new column and assigning it to new column 

insur1 = insur.iloc[:,0:22]
insur1.head()

# Aggregate mean of each cluster
insur1.iloc[:,0:22].groupby(insur.clust).mean() 

