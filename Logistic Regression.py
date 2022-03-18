# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 20:38:11 2022

@author: LENOVO
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data Pre-Processing
affairs = pd.read_csv('D:\Data Set/Affairs.csv')
affairs

affairs.drop(columns='Unnamed: 0', axis=1, inplace=True	)

#Binary binning or Discretization
affairs['naffairs'] = pd.cut(affairs['naffairs'], bins=[min(affairs.naffairs) - 1, affairs['naffairs'].mean(), max(affairs.naffairs)], labels=["affair","no affair"])
affairs.head()

# extracting independent & dependent variable
x = affairs.iloc[:,2:]
y = affairs.iloc[:,1]

# splitting the dataset into training & test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)

# fitting model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

# predicting the results
y_pred = classifier.predict(x_test)
y_pred

# Accuracy of results
from sklearn.metrics import confusion_matrix, accuracy_score

confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)


#Visualizing the training set result  
from matplotlib.colors import ListedColormap  
x_set, y_set = x_train, y_train  
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),  
np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  

plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
alpha = 0.75, cmap = ListedColormap(('purple','green' )))  
plt.xlim(x1.min(), x1.max())  
plt.ylim(x2.min(), x2.max())  
for i, j in enumerate(np.unique(y_set)):  
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
        c = ListedColormap(('purple', 'green')(i)), label = j) 
     
plt.title('Logistic Regression (Training set)')  
plt.xlabel('Age')  
plt.ylabel('Estimated Salary')  
plt.legend()  
plt.show()  


# ========================================*****************============


# 2. Objective:  to develop a strategy by analyzing their customer data. For this, data like age, location, time of activity, etc. has been collected to determine whether a user will click on an ad or not. Perform Logistic Regression on the given data to predict whether a user will click on an ad or not. 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
import statsmodels.formula.api as sm
import numpy as np

#Importing Data
ad = pd.read_csv(r"D:\Data Set\advertising.csv")
ad.head()
ad.columns

ad.isna().sum() #cheking null values
#renaming columns
ad.rename(columns = {'Daily_Time_ Spent _on_Site':'daily_time_spent_on_site','Daily Internet Usage':'daily_internet_usage'}, inplace=True)
# dropiing columns
ad.drop(columns=['Ad_Topic_Line', 'City','Country','Timestamp'], axis=1, inplace=True)


# Model building 
# import statsmodels.formula.api as sm
logit_model = sm.logit('Clicked_on_Ad ~ daily_time_spent_on_site + Age + Area_Income + daily_internet_usage + Male', data = ad).fit()

    
#summary
logit_model.summary2() # for AIC
logit_model.summary()

pred = logit_model.predict(ad.iloc[:,0:5])  

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(ad['Clicked_on_Ad'], pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold

import pylab as plt

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = plt.subplots()
plt.plot(roc['tpr'], color = 'red')
plt.plot(roc['1-fpr'], color = 'blue')
plt.xlabel('1-False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

# filling all the cells with zeroes
ad["pred"] = np.zeros(1000)
# taking threshold value and above the prob value will be treated as correct value 
ad.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(ad["pred"], ad["Clicked_on_Ad"])
classification

### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(ad, test_size = 0.3) # 30% test data

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('Clicked_on_Ad ~ daily_time_spent_on_site + Age + Area_Income + daily_internet_usage + Male', data = train_data).fit()

#summary
model.summary2() # for AIC
model.summary()

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Clicked on Ad
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(300)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['Clicked_on_Ad'])
confusion_matrix

accuracy_test = (150 + 142)/(300) 
accuracy_test # 97.33 %

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["Clicked_on_Ad"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["Clicked_on_Ad"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test

# prediction on train data
train_pred = model.predict(train_data.iloc[ :, 0:5 ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(700)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['Clicked_on_Ad'])
confusion_matrx

accuracy_train = (344 + 335)/(700)
print(accuracy_train)  # 97 %


# ================== ********************** =================================
# 3. to determine whether a user will click on an ad or not using logistic regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Data Pre-Processing
election_data = pd.read_csv('D:\Data Set/election_data.csv')
election_data.columns

election_data.drop(columns=['Year'], axis=1, inplace=True	)
election_data.rename(columns = {'Amount Spent':'amount_spent','Popularity Rank':'popularity_rank'}, inplace=True)

election_data.isnull().sum() # checking null values

# Median imputation
result_value = election_data.Result.median()
result_value
election_data.Result = election_data.Result.fillna(result_value)
election_data.Result.isna().sum()

spent_value = election_data.amount_spent.median()
spent_value
election_data.amount_spent = election_data.amount_spent.fillna(spent_value)
election_data.amount_spent.isna().sum()

rank_value = election_data.popularity_rank.median()
rank_value
election_data.popularity_rank = election_data.popularity_rank.fillna(rank_value)
election_data.popularity_rank.isna().sum()

# Model building 
import statsmodels.formula.api as sm
logit_model1 = sm.logit('Result ~ popularity_rank + amount_spent', data = election_data).fit()
    
#summary
logit_model1.summary2() # for AIC
logit_model1.summary()

pred = logit_model1.predict(election_data.iloc[:,0:])  

from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(election_data['Result'], pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold

import pylab as plt

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = plt.subplots()
plt.plot(roc['tpr'], color = 'red')
plt.plot(roc['1-fpr'], color = 'blue')
plt.xlabel('1-False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

# filling all the cells with zeroes
election_data["pred"] = np.zeros(11)
# taking threshold value and above the prob value will be treated as correct value 
election_data.loc[pred > optimal_threshold, "pred"] = 1

# classification report
from sklearn.metrics import classification_report
classification = classification_report(election_data["pred"], election_data["Result"])
classification

### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(election_data, test_size = 0.3) # 30% test data

# Model building 
import statsmodels.formula.api as sm
model2 = sm.logit('Result ~ amount_spent + popularity_rank', data = train_data).fit()

#summary
model2.summary2() # for AIC
model2.summary()

# Prediction on Test data set
test_pred = model2.predict(test_data)

# Creating new column for storing predicted class of Clicked on Ad
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(4)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['Result'])
confusion_matrix

accuracy_test = (2 + 2)/(4) 
accuracy_test # 

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["Result"])
classification_test

#ROC CURVE AND AUC

from sklearn import metrics
from sklearn.metrics import roc_curve, auc
fpr, tpr, threshold = metrics.roc_curve(test_data["Result"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test


# prediction on train data
train_pred = model2.predict(train_data.iloc[ :, 0:5 ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(7)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['Result'])
confusion_matrx

accuracy_train = (2 + 4)/(5)
print(accuracy_train)  

# =========================== ******************************** ==================

# 4. It is vital for banks that customers put in long term fixed deposits as they use it to pay interest to customers and it is not viable to ask every customer if they will put in a long-term deposit or not. So, build a Logistic Regression model to predict whether a customer will put in a long-term fixed deposit or not based on the different variables given in the data. The output variable in the dataset is Y which is binary. Snapshot of the dataset is given below.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# Data Pre-Processing
bank_data = pd.read_csv('D:\Data Set/bank_data.csv')
bank_data.columns

# bank_data.drop(columns=['contact','poutcome'], axis=1, inplace=True	)
# election_data.rename(columns = {'Amount Spent':'amount_spent','Popularity Rank':'popularity_rank'}, inplace=True)

# dummy variaables

pd.get_dummies(bank_data)
bank_data.isnull().sum() # checking null values

# extracting independent & dependent variable
x = bank_data.iloc[:,0:31]
y = bank_data.iloc[:,31]

# splitting the dataset into training & test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)

# fitting model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

# predicting the results
y_pred = classifier.predict(x_test)
y_pred

# Accuracy of results
from sklearn.metrics import confusion_matrix, accuracy_score

confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)  # 89 %


