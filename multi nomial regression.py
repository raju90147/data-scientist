# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 21:59:34 2022

Name: _RAJU BOTTA____________ Batch ID: 05102021___________
Topic: Multinomial Regression. 

"""

# 1.  A University would like to effectively classify their students based on the program they are enrolled in. Perform multinomial regression on the given dataset and provide insights (in the documentation).

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns

mdata = pd.read_csv('D:\Data Set\mdata.csv')
mdata.columns
mdata.drop(columns=['Unnamed: 0','id'], inplace=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
mdata['female'] = le.fit_transform(mdata['female'])
mdata['ses'] = le.fit_transform(mdata['ses'])
mdata['schtyp'] = le.fit_transform(mdata['schtyp'])
mdata['honors'] = le.fit_transform(mdata['honors'])

# Box plot of independent variable distribution for each category of choice
sns.boxplot(x='prog', y=mdata['ses'], data = mdata)
sns.boxplot(x='prog', y=mdata['female'], data=mdata)
sns.boxplot(x='prog', y=mdata['schtyp'], data=mdata)
sns.boxplot(x='honors', y=mdata['honors'], data=mdata)

# scatter plot for each categorical program of mdata

sns.stripplot(x='prog', y=mdata.ses, jitter=True, data=mdata)
sns.stripplot(x='prog', y=mdata.schtyp, jitter=True, data=mdata)
sns.stripplot(x='prog', y=mdata.honors, jitter=True, data=mdata)
sns.stripplot(x='prog', y=mdata.female, jitter=True, data=mdata)

# Scatter plot between each possible pair of independent variable & also histogram for each independent variable

sns.pairplot(mdata)
sns.pairplot(mdata, hue='prog') # with showing the category of each car choice in the scatter plot

# Correlation values b/w each independent features
mdata.corr()
x=mdata.loc[:,mdata.columns!='prog']
y=mdata['prog']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

# Multinomial option is supported by the 'lbfgs' & 'newton_cg' solvers

model = LogisticRegression(multi_class='multinomial', solver='lbfgs').fit(x_train, y_train)

# Model Prediction
test_predict1 = model.predict(x_test)
test_predict1

# Test Prediction
accuracy_score(y_test, test_predict1)

# ============================== ****************** ======================


#2.  Objective: You work for a consumer finance company which specializes in lending loans to urban customers. When the company receives a loan application, the company has to make a decision for loan approval based on the applicant’s profile. Two types of risks are associated with the bank’s decision: 

loan = pd.read_csv('D:\Data Set\loan.csv')
loan.columns
loan.drop(columns=['id','member_id','next_pymnt_d','emp_title','emp_length','issue_d','url','pymnt_plan','desc','purpose','title','zip_code','mths_since_last_major_derog','annual_inc_joint','dti_joint','verification_status_joint','tot_coll_amt','tot_cur_bal','open_acc_6m','open_il_6m','open_il_12m','open_il_24m','mths_since_rcnt_il','total_bal_il','il_util','open_rv_12m','open_rv_24m','max_bal_bc','all_util','total_rev_hi_lim','inq_fi','addr_state','earliest_cr_line','last_pymnt_d','last_credit_pull_d','dti_joint','verification_status_joint'], axis=1, inplace=True)

# label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
loan['term'] = le.fit_transform(loan['term']) 
loan['grade'] = le.fit_transform(loan['grade']) 
loan['sub_grade'] = le.fit_transform(loan['sub_grade']) 
loan['home_ownership'] = le.fit_transform(loan['home_ownership']) 
loan['verification_status'] = le.fit_transform(loan['verification_status']) 
loan['loan_status'] = le.fit_transform(loan['loan_status']) 
loan['application_type'] = le.fit_transform(loan['application_type']) 
loan['initial_list_status'] = le.fit_transform(loan['initial_list_status']) 

# basic EDA
loan.describe()
loan.loan_status.value_counts()

# Checking null values
loan.isnull().sum()

tax_med = loan.tax_liens.mean()
loan.tax_liens = loan.tax_liens.fillna(tax_med)

# MEAN OR MODE Imputation
pub_med = loan.pub_rec_bankruptcies.median()
loan.pub_rec_bankruptcies = loan.pub_rec_bankruptcies.fillna(pub_med)

charge_med = loan.chargeoff_within_12_mths.median()
loan.chargeoff_within_12_mths = loan.chargeoff_within_12_mths.fillna(charge_med)

collect_med = loan.collections_12_mths_ex_med.median()
loan.collections_12_mths_ex_med = loan.collections_12_mths_ex_med.fillna(collect_med)

revol_med = loan.revol_util.median()
loan.revol_util = loan.revol_util.fillna(revol_med)

mths_med = loan.mths_since_last_record.median()
loan.mths_since_last_record = loan.mths_since_last_record.fillna(mths_med)

mth_med = loan.mths_since_last_delinq.median()
loan.mths_since_last_delinq = loan.mths_since_last_delinq.fillna(mths_med)


# converting percentage column into float

loan['revol_util'] = loan['revol_util'].str.rstrip('%').astype('float') / 100.0
loan['int_rate'] = loan['int_rate'].str.rstrip('%').astype('float') / 100.0

 
loan.columns
# Scatter plot for each categorical choice of car
sns.stripplot(x = "loan_status", y = loan['recoveries'], jitter = True, data = loan)
sns.stripplot(x = "loan_status", y = loan['annual_inc'], jitter = True, data = loan)
sns.stripplot(x = "loan_status", y = loan['chargeoff_within_12_mths'], jitter = True, data = loan)
sns.stripplot(x = "loan_status", y = loan['tax_liens'], jitter = True, data = loan)

# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(loan) # Normal

sns.pairplot(loan, hue = "loan_status") # With showing the category of each car choice in the scatter plot
 
# Correlation values between each independent features
loan.corr()
loan.columns

x = loan.loc[:,loan.columns!='loan_status']
y = loan['loan_status']

from sklearn.model_selection import train_test_split
train_x, train_y, test_x, test_y = train_test_split(x,y, test_size = 0.2)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train_x, train_y)

test_predict = model.predict(test_x) # Test predictions

# Test accuracy 
accuracy_score(test_predict, test_y)

train_predict = model.predict(train_x) # Train predictions 
# Train accuracy 
accuracy_score(train_y, test_predict) 

     