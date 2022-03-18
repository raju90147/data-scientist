# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:05:37 2022

@author: LENOVO
"""
'''
Topic: Data Pre-Processing


Problem Statement:  
Most of the datasets have extreme values or exceptions in their observations. These values affect the predictions (Accuracy) of the model in one way or the other, removing these values is not a very good option. For these types of scenarios, we have various techniques to treat such values. 
Refer: https://360digitmg.com/mindmap-data-science

1.	Prepare the dataset by performing the preprocessing techniques, to treat the outliers.

 

Hints:
For each assignment, the solution should be submitted in the below format
1.	Work on each feature to create a data dictionary as displayed in the image displayed below: 
2.	Hint: Boston dataset is publicly available. Refer to Boston.csv file.
3.	Research and perform all possible steps for obtaining solution
4.	All the codes (executable programs) should execute without errors
5.	Code modularization should be followed
6.	Each line of code should have comments explaining the logic and why you are using that function
7.	Detailed explanation of your approach is mandatory
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r'D:\DataSets\boston_data.csv')
df

df.shape

#let's find outliers, and data is normally distributed or not

sns.boxplot(df.crim)
sns.boxplot(df.zn)
sns.boxplot(df.indus)
sns.boxplot(df.nox) # no outliers
sns.boxplot(df.rm) #No outliers
sns.boxplot(df.age) #No Outliers
sns.boxplot(df.dis) 
sns.boxplot(df.rad) 
sns.boxplot(df.tax) #no outliers
sns.boxplot(df.ptratio) 
sns.boxplot(df.black)
sns.boxplot(df.lstat) 
sns.boxplot(df.medv)

# RRR Technique to treat outliers
#outlier treatment for crim

IQR = df['crim'].quantile(0.75)-df['crim'].quantile(0.25)
lower_limit = df['crim'].quantile(0.25)-(IQR)*1.5
upper_limit = df['crim'].quantile(0.75)+(IQR)*1.5

#1 remove or Trimming technic 

outliers_df = np.where(df['crim']>upper_limit,True, np.where(df['crim']<lower_limit,True, False))
df_trimmed = df.loc[~(outliers_df),]
df_trimmed.shape

#explore outliers in the trimmed data set

sns.boxplot(df_trimmed.crim)
plt.title('boxplot')

# 2. replace

df['df_replace'] = pd.DataFrame(np.where(df['crim']>upper_limit,upper_limit, np.where(df['crim']<lower_limit,lower_limit,df['crim'])))

sns.boxplot(df['df_replace']) 
plt.title('box plot')
plt.show()

#3. retain or winsorization

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr',tail='both',fold=1.5, variables=('crim'))
df_t = winsor.fit_transform(df[['crim']])
df_t

sns.boxplot(df_t.crim)
plt.title('Box Plot')
plt.show

#outlier treatment for zn
#Finding IQR for zn

IQR1 = df['zn'].quantile(0.75)-df['zn'].quantile(0.25)
lower_limit1 = df['zn'].quantile(0.25)-(IQR1)*1.5
upper_limit1 = df['zn'].quantile(0.75)+(IQR1)*1.5

#remove
outliers_df1 = np.where(df['zn']>upper_limit1,True, np.where(df['zn']<lower_limit1,True, False))
df_trimmed1 = df.loc[~(outliers_df1),]
df_trimmed1.shape
sns.boxplot(df_trimmed1.zn)
plt.title('boxplot')

#replace
df['df_replace1'] = pd.DataFrame(np.where(df['zn']>upper_limit1,upper_limit1, np.where(df['zn']<lower_limit1,lower_limit1,df['zn'])))

sns.boxplot(df['df_replace1']) 
plt.title('box plot')
plt.show()

#retain or winsorize
winsor1 = Winsorizer(capping_method='iqr',tail='both',fold=1.5, variables=('zn'))
df_t = winsor1.fit_transform(df[['zn']])
df_t

sns.boxplot(df_t.zn)
plt.title('Box Plot')
plt.show

#outlier treatment for indus
#IQR for indus
IQR2 = df['indus'].quantile(0.75)-df['indus'].quantile(0.25)
lower_limit2 = df['indus'].quantile(0.25)-(IQR2)*1.5
upper_limit2= df['indus'].quantile(0.75)+(IQR2)*1.5

#remove
outliers_df2 = np.where(df['indus']>upper_limit2,True, np.where(df['indus']<lower_limit2,True, False))
df_trimmed2 = df.loc[~(outliers_df2),]
df_trimmed2.shape
sns.boxplot(df_trimmed2.indus)
plt.title('boxplot')

#replace
df['df_replace2'] = pd.DataFrame(np.where(df['indus']>upper_limit2,upper_limit2, np.where(df['indus']<lower_limit2,lower_limit2,df['indus'])))

sns.boxplot(df['df_replace2']) 
plt.title('box plot')
plt.show()

#retain or winsorize
winsor2 = Winsorizer(capping_method='iqr',tail='both',fold=1.5, variables=('indus'))
df_t = winsor2.fit_transform(df[['indus']])
df_t

sns.boxplot(df_t.indus)
plt.title('Box Plot')
plt.show

#outlier treatment for dis
#IQR
IQR3 = df['dis'].quantile(0.75)-df['dis'].quantile(0.25)
lower_limit3 = df['dis'].quantile(0.25)-(IQR3)*1.5
upper_limit3= df['dis'].quantile(0.75)+(IQR3)*1.5

#Remove
outliers_df3 = np.where(df['dis']>upper_limit3,True, np.where(df['dis']<lower_limit3,True, False))
df_trimmed3 = df.loc[~(outliers_df3),]
df_trimmed3.shape
sns.boxplot(df_trimmed3.dis)
plt.title('boxplot')

#replace
df['df_replace3'] = pd.DataFrame(np.where(df['dis']>upper_limit3,upper_limit3, np.where(df['dis']<lower_limit3,lower_limit3,df['dis'])))

sns.boxplot(df['df_replace3']) 

#retain or winsorize
winsor3 = Winsorizer(capping_method='iqr',tail='both',fold=1.5, variables=('dis'))
df_t = winsor3.fit_transform(df[['dis']])
df_t

sns.boxplot(df_t.dis)
plt.title('Box Plot')
plt.show

#outlier treatment for rad
#IQR

IQR4 = df['rad'].quantile(0.75)-df['rad'].quantile(0.25)
lower_limit4 = df['rad'].quantile(0.25)-(IQR4)*1.5
upper_limit4= df['rad'].quantile(0.75)+(IQR4)*1.5

#remove
outliers_df4 = np.where(df['rad']>upper_limit4,True, np.where(df['rad']<lower_limit4,True, False))
df_trimmed4 = df.loc[~(outliers_df4),]
df_trimmed4.shape
sns.boxplot(df_trimmed4.rad)
plt.title('boxplot')

#replace
df['df_replace4'] = pd.DataFrame(np.where(df['rad']>upper_limit4,upper_limit4, np.where(df['rad']<lower_limit4,lower_limit4,df['rad'])))

sns.boxplot(df['df_replace4']) 

#retain or winsorize for rad
winsor4 = Winsorizer(capping_method='iqr',tail='both',fold=1.5, variables=('rad'))
df_t = winsor4.fit_transform(df[['rad']])
df_t

sns.boxplot(df_t.rad)
plt.title('Box Plot')
plt.show

#outlier treatment ptratio
#IQR
IQR5 = df['ptratio'].quantile(0.75)-df['ptratio'].quantile(0.25)
lower_limit5 = df['ptratio'].quantile(0.25)-(IQR5)*1.5
upper_limit5= df['ptratio'].quantile(0.75)+(IQR5)*1.5

#remove
outliers_df5 = np.where(df['ptratio']>upper_limit5,True, np.where(df['ptratio']<lower_limit5,True, False))
df_trimmed5 = df.loc[~(outliers_df5),]
df_trimmed5.shape
sns.boxplot(df_trimmed5.ptratio)
plt.title('boxplot')

#replace
df['df_replace5'] = pd.DataFrame(np.where(df['ptratio']>upper_limit5,upper_limit5, np.where(df['ptratio']<lower_limit5,lower_limit5,df['ptratio'])))

sns.boxplot(df['df_replace5']) 

#retain or winsorize for rad
winsor5 = Winsorizer(capping_method='iqr',tail='both',fold=1.5, variables=('ptratio'))
df_t = winsor5.fit_transform(df[['ptratio']])
df_t

sns.boxplot(df_t.ptratio)
plt.title('Box Plot')
plt.show

#outlier treatment for black
#IQR

IQR6 = df['black'].quantile(0.75)-df['black'].quantile(0.25)
lower_limit6 = df['black'].quantile(0.25)-(IQR)*1.5
upper_limit6 = df['black'].quantile(0.75)+(IQR)*1.5

#remove
outliers_df6 = np.where(df['black']>upper_limit6,True, np.where(df['black']<lower_limit6,True, False))
df_trimmed6 = df.loc[~(outliers_df6),]
df_trimmed6.shape
sns.boxplot(df_trimmed6.black)
plt.title('boxplot')

#replace
df['df_replace6'] = pd.DataFrame(np.where(df['black']>upper_limit6,upper_limit6, np.where(df['black']<lower_limit6,lower_limit6,df['black'])))

sns.boxplot(df['df_replace6']) 

#retain or winsorize for rad
winsor6 = Winsorizer(capping_method='iqr',tail='both',fold=1.5, variables=('black'))
df_t = winsor6.fit_transform(df[['black']])
df_t

sns.boxplot(df_t.black)
plt.title('Box Plot')
plt.show

#outlier treatment for black
#IQR

IQR7 = df['istat'].quantile(0.75)-df['Istat'].quantile(0.25)
lower_limit7 = df['Istat'].quantile(0.25)-(IQR)*1.5
upper_limit7 = df['Istat'].quantile(0.75)+(IQR)*1.5

#remove
outliers_df7 = np.where(df['Istat']>upper_limit7,True, np.where(df['Istat']<lower_limit7,True, False))
df_trimmed7 = df.loc[~(outliers_df7),]
df_trimmed7.shape
sns.boxplot(df_trimmed7.black)
plt.title('boxplot')

#replace
df['df_replace7'] = pd.DataFrame(np.where(df['Istat']>upper_limit7,upper_limit7, np.where(df['Istat']<lower_limit7,lower_limit7,df['Istat'])))

sns.boxplot(df['df_replace7']) 

#retain or winsorize for rad
winsor7 = Winsorizer(capping_method='iqr',tail='both',fold=1.5, variables=('Istat'))
df_t = winsor7.fit_transform(df[['Istat']])
df_t

sns.boxplot(df_t.Istat)
plt.title('Box Plot')
plt.show

#outlier treatment for medv
#IQR

IQR8 = df['medv'].quantile(0.75)-df['medv'].quantile(0.25)
lower_limit8 = df['medv'].quantile(0.25)-(IQR)*1.5
upper_limit8 = df['medv'].quantile(0.75)+(IQR)*1.5

#remove
outliers_df8 = np.where(df['medv']>upper_limit8,True, np.where(df['medv']<lower_limit8,True, False))
df_trimmed8 = df.loc[~(outliers_df8),]
df_trimmed8.shape
sns.boxplot(df_trimmed8.black)
plt.title('boxplot')

#replace
df['df_replace8'] = pd.DataFrame(np.where(df['medv']>upper_limit8,upper_limit8, np.where(df['medv']<lower_limit8,lower_limit8,df['medv'])))

sns.boxplot(df['df_replace8']) 

#retain or winsorize for rad
winsor8 = Winsorizer(capping_method='iqr',tail='both',fold=1.5, variables=('medv'))
df_t = winsor8.fit_transform(df[['medv']])
df_t

sns.boxplot(df_t.medv)
plt.title('Box Plot')
plt.show

#outlier treatment for lstat
#IQR

IQR9 = df['lstat'].quantile(0.75)-df['lstat'].quantile(0.25)
lower_limit9 = df['lstat'].quantile(0.25)-(IQR)*1.5
upper_limit9 = df['lstat'].quantile(0.75)+(IQR)*1.5

#remove
outliers_df9 = np.where(df['lstat']>upper_limit9,True, np.where(df['lstat']<lower_limit9,True, False))
df_trimmed9 = df.loc[~(outliers_df9),]
df_trimmed9.shape
sns.boxplot(df_trimmed9.lstat)
plt.title('boxplot')

#replace
df['df_replace9'] = pd.DataFrame(np.where(df['lstat']>upper_limit9,upper_limit9, np.where(df['lstat']<lower_limit9,lower_limit9,df['lstat'])))

sns.boxplot(df['df_replace9']) 

#retain or winsorize for rad
winsor9 = Winsorizer(capping_method='iqr',tail='both',fold=1.5, variables=('lstat'))
df_t = winsor9.fit_transform(df[['lstat']])
df_t

sns.boxplot(df_t.lstat)
plt.title('Box Plot')
plt.show

# Conclusion: In the above problem outlier analysis has done using 3 technics remove, replace and retain or winsorize based on IQR (Inter Quartile Range).


