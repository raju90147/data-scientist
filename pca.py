# -*- coding: utf-8 -*-
"""
Name: _BOTTA RAJU____________ Batch ID: _28-02-2021__________	
Topic: Principal Component Analysis

"""
# 1. Heart discease

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

heart = pd.read_csv('D:\Data Set\heart disease.csv')
heart.describe()
heart.info()

  # EDA
  # Univariate Analysis
  # Distribution plots
  
plt.hist(heart['chol'], 10)  

plt.boxplot(heart['age'])  
  
#Bivariate Analysis  
plt.bar(height=heart.age, x = heart['chol'])
 
plt.scatter(heart.age, heart['chol'])

 
  # Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(heart.iloc[:, 0:])
df_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));
plt.title('Hierarchical Clustering Dendrogram');
plt.xlabel('age');
plt.ylabel('cholestorol')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

'''
The vertical line with maximum distance is the blue line and hence we can decide the line that cut the dendrogram.

We have 4 clusters as this line cuts the dendrogram at 4 points. Let’s now apply hierarchical clustering for 4 clusters:


Conclusion: Applied AgglomerativeClustering choosing 5 as clusters from the above dendrogram.
'''

# Now applying AgglomerativeClustering choosing 4 as clusters from the above dendrogram

from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = 'uclidean').fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

heart['clust'] = cluster_labels # creating a new column and assigning it to new column 

heart = heart.iloc[:,0:]
heart.head()

# Aggregate mean of each cluster

heart.iloc[:, 2:].groupby(heart.clust).mean()

    # K-Means Clustering
         
###### scree plot or elbow curve ############

TWSS = []
k = list(range(1, 15))

from sklearn.cluster import	KMeans

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
  
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")


# Selecting  clusters from the above scree plot which is the optimum number of clusters 

model = KMeans(n_clusters = 4)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
heart['clust'] = mb # creating a  new column and assigning it to new column 

heart.head()
df_norm.head()

heart = heart.iloc[:,0:]
heart.head()

heart.iloc[:, 0:].groupby(heart.clust).mean()
               
  # PCA 

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

#Consider only numeric data

heart.data = heart.iloc[:,0:]

#normalizing the numerical data

heart_normal = scale(heart.data)
heart_normal

pca = PCA(n_components=14)
pca_values = pca.fit_transform(heart_normal)

#The amount of variance that each pca explains 

var = pca.explained_variance_ratio_
var

#PCA Weights

pca.components_
pca.components_[0]

#Cumulative variance

var1 = np.cumsum(np.round(var,decimals=4)*100)
var1

#Variance plot for PCA Components obtained 

plt.plot(var1, color ='red')

 
#PCA Scores

pca_values

pca_data = pd.DataFrame(pca_values)
pca_data.columns = 'comp0', 'comp1', 'comp2', 'comp3','comp4','comp5','comp6','comp7','comp8','comp9','comp10','comp11','comp12','comp13'

final = pd.concat([heart,pca_data.iloc[:,0:3]], axis=1)

#Scatter Diagram

import matplotlib.pylab as plt

ax = final.plot(x='comp0', y='comp1', kind='scatter',figsize=(12,8))
final[['comp0', 'comp1', 'slope']].apply(lambda x: ax.text(*x), axis=1)


# The above plot shows relation between comp0 and comp1 
 
# =================== ******************* =================

# wine dataset


wine = pd.read_csv('D:\Data Set\wine.csv')
wine.describe()
wine.info()

  # EDA
  #Univariate Analysis

plt.hist(wine['Alcohol'], 10)  

#Bivariate Analysis  

plt.bar(height=wine.Alcohol, x = wine['Type'])
 
plt.boxplot(wine['Alcohol'])  

  # Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(wine.iloc[:, 0:])
df_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));
plt.title('Hierarchical Clustering Dendrogram');
plt.xlabel('Alcohol');
plt.ylabel('Ash')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

''' 
We have 5 clusters as this line cuts the dendrogram at 5 points. Let’s now apply hierarchical clustering for 5 clusters:

Conclusion: Applied AgglomerativeClustering choosing 5 as clusters from the above dendrogram
'''
# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram

from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 5, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

wine['clust'] = cluster_labels # creating a new column and assigning it to new column 

wine = wine.iloc[:,0:]
wine.head()

# Aggregate mean of each cluster
wine.iloc[:, 2:].groupby(wine.clust).mean()

  
# K-Means
        
###### scree plot or elbow curve ############

TWSS = []
k = list(range(1, 15))

from sklearn.cluster import	KMeans

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
  
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

 

# Selecting  clusters from the above scree plot which is the optimum number of clusters 

model = KMeans(n_clusters = 3)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
wine['clust'] = mb # creating a  new column and assigning it to new column 

wine.head()
df_norm.head()

wine = wine.iloc[:,0:]
wine.head()

wine.iloc[:, 0:].groupby(wine.clust).mean()
         
  # PCA 

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

#Consider only numeric data

wine.data = wine.iloc[:,0:]

#normalizing the numerical data

wine_normal = scale(wine.data)
wine_normal

pca = PCA(n_components=15)
pca_values = pca.fit_transform(wine_normal)

#The amount of variance that each pca explains 

var = pca.explained_variance_ratio_
var

#PCA Weights

pca.components_
pca.components_[0]

#Cumulative variance

var1 = np.cumsum(np.round(var,decimals=4)*100)
var1

#Variance plot for PCA Components obtained 

plt.plot(var1, color ='red')

#PCA Scores

pca_values

pca_data = pd.DataFrame(pca_values)
pca_data.columns = 'comp0', 'comp1', 'comp2', 'comp3','comp4','comp5','comp6','comp7','comp8','comp9','comp10','comp11','comp12','comp13','comp14'

final = pd.concat([heart,pca_data.iloc[:,0:7]], axis=1)
 
#Scatter Diagram

import matplotlib.pylab as plt

ax = final.plot(x='comp0', y='comp1', kind='scatter',figsize=(12,8))
final[['comp0', 'comp1', 'slope']].apply(lambda x: ax.text(*x), axis=1)

 
