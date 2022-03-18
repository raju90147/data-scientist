# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 12:25:17 2022

Name: _BOTTA____________ Batch ID: __05102021_________
Topic: Decision Tree and Random Forest

"""

# 1. Company Data

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

tree = pd.read_csv(r'D:\Data Set\Company_Data.csv')

# to convert numeric into categorical or discretization

tree['Sales'] = pd.cut(tree['Sales'], bins=[min(tree.Sales) - 1, tree.Sales.mean(), max(tree.Sales)], labels=["Low","High"])

tree.info
tree.head()
tree.columns

from sklearn.preprocessing import LabelEncoder

#label encoding to convert categorical to binary
le = LabelEncoder()
tree["ShelveLoc"] = le.fit_transform(tree["ShelveLoc"])
tree["Urban"] = le.fit_transform(tree["Urban"])
tree["US"] = le.fit_transform(tree["US"])


# Feature metrix & dependent vector

x = tree.iloc[:,2:12] #predictors
y = tree['Sales'] # target

sns.countplot(y)

# Data Splitting

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion='entropy')
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True)
accuracy_score(y_test, y_pred)

# Prediction on Test Data
preds = classifier.predict(x_test)
pd.crosstab(y_test, preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == y_test) # Test Data Accuracy 

# Prediction on Train Data
preds = classifier.predict(x_train)
pd.crosstab(y_train, preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == y_train) # Train Data Accuracy

# Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier 

rfc = RandomForestClassifier(n_estimators=3, random_state=0)
rfc.fit(x_train, y_train)

# Predictions on testing data
y_pred = rfc.predict(x_test)
y_pred

# accuracy 
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True)
accuracy_score(y_test, y_pred)


# Prediction on Test Data
preds = classifier.predict(x_test)
pd.crosstab(y_test, preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == y_test) # Test Data Accuracy 

# Prediction on Train Data
preds = classifier.predict(x_train)
pd.crosstab(y_train, preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == y_train) # Train Data Accuracy

# ---------------------------------------------------------------



# 2. Objective: Divide the diabetes data into train and test datasets and build a Random Forest and Decision Tree model with Outcome as the output variable. 

diabetes = pd.read_csv('D:\Data Set\Diabetes.csv')
diabetes.head()
#diabetes(columns = {'Class variable':'class'}, axis=1, inplace = True)

diabetes.columns
diabetes.isna().sum()

from sklearn.preprocessing import LabelEncoder

#label encoding to convert categorical to binary
le = LabelEncoder()
diabetes[" Class variable"] = le.fit_transform(diabetes[" Class variable"])

# Feature metrix & dependent vector

x = diabetes.iloc[:,1:8] #predictors
y = diabetes[' Class variable'] # target

sns.countplot(y)

# Data Splitting

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Train the Regression DT
from sklearn import tree

regtree = tree.DecisionTreeRegressor(max_depth = 3)
regtree.fit(x_train, y_train)
y_pred = regtree.predict(x_test)

# Prediction
test_pred = regtree.predict(x_test)
train_pred = regtree.predict(x_train)

# Measuring accuracy
from sklearn.metrics import mean_squared_error, r2_score


# Error on test dataset
mean_squared_error(y_test, y_pred)
r2_score(y_test, y_pred)

# Error on train dataset
mean_squared_error(y_train, train_pred)
r2_score(y_train, train_pred)
regtree2 = tree.DecisionTreeRegressor(min_samples_split = 3)
regtree2.fit(x_train, y_train)

# Prediction
test_pred2 = regtree2.predict(x_test)
train_pred2 = regtree2.predict(x_train)

# Error on test dataset
mean_squared_error(y_test, test_pred2)
r2_score(y_test, test_pred2)

# Error on train dataset
mean_squared_error(y_train, train_pred2)
r2_score(y_train, train_pred2)

########### Decistion tree regression prune
## Minimum observations at the leaf node approach
regtree3 = tree.DecisionTreeRegressor(min_samples_leaf = 3)
regtree3.fit(x_train, y_train)

# Prediction
test_pred3 = regtree3.predict(x_test)
train_pred3 = regtree3.predict(x_train)

# measure of error on test dataset
mean_squared_error(y_test, test_pred3)
r2_score(y_test, test_pred3)

# measure of error on train dataset
mean_squared_error(y_train, train_pred3)
r2_score(y_train, train_pred3)


# Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor 

rfc = RandomForestRegressor(n_estimators=3, random_state=0)
rfc.fit(x_train, y_train)

# Predictions on testing data
y_pred = rfc.predict(x_test)
y_pred

# accuracy 
from sklearn.metrics import mean_squared_error, r2_score


# Error on test dataset
mean_squared_error(y_test, y_pred)
r2_score(y_test, y_pred)

# Error on train dataset
mean_squared_error(y_train, y_pred)
r2_score(y_train, train_pred)

# ----------------------------------================= ******************

# 3.Objective:	Build a Decision Tree & Random Forest model on the fraud data. Treat those who have taxable_income <= 30000 as Risky and others as Good (discretize the taxable 

fraud_check = pd.read_csv('D:\Data Set\Fraud_check.csv')
fraud_check.head()
fraud_check.columns
fraud_check.isna().sum()
fraud_check['Taxable.Income']

# Discretization  
fraud_check["Taxable.Income"]=pd.cut(fraud_check["Taxable.Income"],bins=[0,30000,100000],labels=["Risky", "Good"])
from sklearn.preprocessing import LabelEncoder
# label encoding to convert category into numerical
lb = LabelEncoder()
fraud_check["Undergrad"] = lb.fit_transform(fraud_check["Undergrad"])
fraud_check["Marital.Status"] = lb.fit_transform(fraud_check["Marital.Status"])
fraud_check["Urban"] = lb.fit_transform(fraud_check["Urban"])


 # Feature metrix & dependent vector

x = fraud_check.drop('Taxable.Income', axis = 1)
y = fraud_check['Taxable.Income']
    
from sklearn.model_selection import train_test_split

x_train,x_test,y_train ,y_test = train_test_split(x,y,test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.tree import DecisionTreeClassifier

classify = DecisionTreeClassifier(criterion='entropy')
classify.fit(x_train, y_train)
y_pred = classify.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True)
accuracy_score(y_test, y_pred)

# Prediction on Test Data
preds = classify.predict(x_test)
pd.crosstab(y_test, preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == y_test) # Test Data Accuracy 

# Prediction on Train Data
preds = classify.predict(x_train)
pd.crosstab(y_train, preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == y_train) # Train Data Accuracy

# Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier 

rfc = RandomForestClassifier(n_estimators=3, random_state=0)
rfc.fit(x_train, y_train)

# Predictions on testing data
y_pred = rfc.predict(x_test)
y_pred

# accuracy 
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True)
accuracy_score(y_test, y_pred)


# Prediction on Test Data
preds = classifier.predict(x_test)
pd.crosstab(y_test, preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == y_test) # Test Data Accuracy 

# Prediction on Train Data
preds = classifier.predict(x_train)
pd.crosstab(y_train, preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == y_train) # Train Data Accuracy



# -----------------------------------                                                                         
#4.  Objective : HR faces the challenge of predicting if the candidate is faking their salary or not. For example, a candidate claims to have 5 years of experience and earns 70,000 per month working as a regional manager. The candidate expects more money than his previous CTC. We need a way to verify their claims (is 70,000 a month working as a regional manager with an experience of 5 years a genuine claim or does he/she make less than that?) Build a Decision Tree and Random Forest model with monthly income as the target variable. 
import pandas as pd  
hr = pd.read_csv('D:\Data Set\HR_DT.csv')
hr.head()                                                                           
hr.columns     
# Missing values
hr.isnull().sum()
# there are no missing values in our dataset

### EDA, Pre-processing
hr.info()
hr.head()
hr['Position of the employee'].value_counts(normalize = True)
# # we see this is ordinal data so we can convert it to numerical position wise, higher position - higher number, lower position - lower number
'''
position_label_mapping = {'Business Analyst': 1, 'Junior Consultant': 2, 'Senior Consultant': 3, 'Manager': 4, 'Country Manager': 5,'Region Manager': 6,
                          'Partner': 7, 'Senior Partner': 8, 'C-level': 9, 'CEO': 10}

hr['Position of the employee'] = hr['Position of the employee'].map(position_label_mapping)
'''
# or Label encoding

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
hr["Position of the employee"] = lb.fit_transform(hr["Position of the employee"])

# Input and Output Split
predictors = hr.iloc[:,:-1]
target = hr[" monthly income of employee"]
# Input and Output Split
# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.25, random_state=0)

from sklearn import tree
regtree = tree.DecisionTreeRegressor()
regtree.fit(x_train, y_train)

# Prediction
test_pred = regtree.predict(x_test)
train_pred = regtree.predict(x_train)

# Measuring accuracy
from sklearn.metrics import mean_squared_error, r2_score
# Error on test dataset
mean_squared_error(y_test, test_pred)

r2_score(y_test, test_pred)
# we get Rsquare of 0.98 on test dataset which is very good

# Error on train dataset
mean_squared_error(y_train, train_pred)

r2_score(y_train, train_pred)
# we get Rsquare of 0.999 which is almost 1 on train dataset

# let us try Random forest
from sklearn.ensemble import RandomForestRegressor
rf_clf = RandomForestRegressor(n_estimators=500, n_jobs=1, random_state=42)
rf_clf.fit(x_train, y_train)

# Error on test dataset
mean_squared_error(y_test, rf_clf.predict(x_test))

r2_score(y_test, rf_clf.predict(x_test))
# we see that our mean squared error goes down
# we also see that our R squared is now 99.15% which has increased
# we will therefore use this model generated by using RandomForrest

# let us now check if candidate is honest or fraud
# first we will have to add his entry to our x_test data

cand = {'Position of the employee': 6, 'no of Years of Experience of employee': 5.0}
x_test_cand = x_test.append(cand, ignore_index = True)
x_test_preds = rf_clf.predict(x_test_cand)
# since it was the last row that we added, we are only interested in predicted value of the last row
x_test_preds[-1]

# Summary: According to our model the candidate appears to be faking his salary

# since the model predicts his salary to be less than 70000.

