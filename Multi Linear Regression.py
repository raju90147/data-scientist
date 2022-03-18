# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 20:19:19 2022

Name: _BOTTA_RAJU___________ Batch ID: _05102021__________
Topic: Multilinear Regression

"""

# Multiple Linear Regression
#1.  Objective: An analytics company has been tasked with the crucial job of finding out what factors affect a startup company and if it will be profitable or not. For this, they have collected some historical data and would like to apply multilinear regression to derive brief insights into their data. Predict profit, given different attributes for various startup companies.

import pandas as pd
import numpy as np

startups = pd.read_csv(r'D:\Data Set\50_Startups.csv')
startups.columns

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 


plt.bar(height = startups['Profit'], x = np.arange(1, 51, 1))
plt.hist(startups.Profit) #histogram
plt.boxplot(startups.Profit) #boxplot

plt.bar(height = startups['Marketing Spend'], x = np.arange(1, 51, 1))
plt.hist(startups['Marketing Spend']) #histogram
plt.boxplot(startups['Marketing Spend']) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=startups['Marketing Spend'], y=startups['Profit'])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(startups['Profit'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(startups['Marketing Spend'], dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(startups.iloc[:, :])
                             
# Correlation matrix 
startups.corr()

# startups.rename(columns = {'R&D Spend':'RS','Marketing Spend':'MS'}, inplace = True)
# we see there exists High collinearity between input variables especially between

# startups = startups.drop('State', axis=1, inplace=True)
startups.rename(columns={'R&D': 'RD'},inplace=True, errors='raise') 

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
startups.columns          
ml1 = smf.ols('RD ~ Administration + Marketing + State + Profit', data = startups).fit() # regression model

# Summary
ml1.summary()
# p-values for WT, VOL are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 76 is showing high influence so we can exclude that entire row

# we see there exists High collinearity between input variables especially between
# [HP & SP], [VOL & WT] so there exists collinearity problem

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
ml2 = smf.ols('Profit ~ Administration + Marketing + State + RD', data = startups).fit() # regression model

# Summary
ml2.summary()
# p-values for WT, VOL are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm
 
sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals

# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_profit = smf.ols('Profit ~ Administration + Marketing + RD + State', data = startups).fit().rsquared  
vif_profit = 1/(1 - rsq_profit) 

rsq_admin = smf.ols('Administration ~ Profit + Marketing + RD + State', data = startups).fit().rsquared  
vif_admin = 1/(1 - rsq_admin)

rsq_marketing = smf.ols('Marketing ~ Profit + Administration + RD + State', data = startups).fit().rsquared  
vif_marketing = 1/(1 - rsq_marketing) 

rsq_rd = smf.ols('RD ~ Profit + Administration + Marketing + State', data = startups).fit().rsquared  
vif_rd = 1/(1 - rsq_rd) 

 
# Storing vif values in a data frame
d1 = {'Variables':['Profit', 'Administration', 'Marketing', 'RD'], 'VIF':[vif_profit, vif_admin, vif_marketing, vif_rd]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As Profit is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('Administration ~ Marketing + RD + State + Profit', data = startups).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(startups)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = res, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)

### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
startups_train, startups_test = train_test_split(startups, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("RD ~ Administration + Marketing + State + Profit", data = startups_train).fit()

# prediction on test data set 
test_pred = model_train.predict(startups_test)

# test residual values 
test_resid = test_pred - startups_test.Marketing
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse

# train_data prediction
train_pred = model_train.predict(startups_train)

# train residual values 
train_resid  = train_pred - startups_train.Marketing
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

# =================== ************ ==========================

 #2.  Objective: With the growing consumption of avocados in the USA, a freelance company would like to do some analysis on the patterns of consumption in different cities and would like to come up with a prediction model for the price of avocados. For this to be implemented, build a prediction model using multilinear regression and provide your insights on it.

price = pd.read_csv('D:\Data Set\Avacado_Price.csv')   
price.columns  

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
price['type'] = le.fit_transform(price['type'])
price['region'] = le.fit_transform(price['region'])
price['year'] = le.fit_transform(price['year'])

x = price.loc[:,price.columns!="AveragePrice"]
y = price['AveragePrice']

# data splitting

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

# Model building
from sklearn.linear_model import LinearRegression

regresser = LinearRegression()
regresser.fit(x_train, y_train)

y_pred = regresser.predict(x_test)
y_pred


# RMSE Values
from sklearn.metrics import mean_squared_error

# Error on test dataset
mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse

# Error on train dataset
mean_squared_error(y_train, y_pred)
rmse1= np.sqrt(mean_squared_error(y_test, y_pred))
rmse1

# ================== **************** ======================= 

# 3. Objective: An online car sales platform would like to improve its customer base and their experience by providing them an easy way to buy and sell cars. For this, they would like an automated model which can predict the price of the car once the user inputs the required factors. Help the business achieve their objective by applying multilinear regression on the given dataset. Please use the below columns for the analysis purpose: price, age_08_04, KM, HP, cc, Doors, Gears, Quarterly_Tax, and Weight.

import pandas as pd
import numpy as np
car_sale = pd.read_csv("D:/Data Set/ToyotaCorolla.csv", dtype=str)   
car_sale.columns  

car_sale = car_sale.drop(columns=['Model','Mfg_Month', 'Mfg_Year', 'Fuel_Type', 'Met_Color', 'Color', 'Automatic', 'Cylinders', 'Mfr_Guarantee',
'BOVAG_Guarantee', 'Guarantee_Period', 'ABS', 'Airbag_1', 'Airbag_2',
 'Airco', 'Automatic_airco', 'Boardcomputer', 'CD_Player',
 'Central_Lock', 'Powered_Windows', 'Power_Steering', 'Radio',
       'Mistlamps', 'Sport_Model', 'Backseat_Divider', 'Metallic_Rim',
       'Radio_cassette', 'Tow_Bar'])
car_sale.head()

x = car_sale.loc[:,car_sale.columns!="Price"]
y = car_sale['Price']

# data splitting

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

# Model building
from sklearn.linear_model import LinearRegression

regresser = LinearRegression()
regresser.fit(x_train, y_train)

y_pred = regresser.predict(x_test)

y_pred

# =============== ********** ====================

# 4. Perform multilinear regression with price as the output variable and document the different RMSE values.
    

computer_data = pd.read_csv('D:\Data Set\Computer_Data.csv')   
computer_data.columns  
computer_data = computer_data.drop(columns='Unnamed: 0')
computer_data

pd.get_dummies(computer_data)

x = computer_data.loc[:,computer_data.columns!="price"]
y = computer_data['price']

# data splitting

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

# Model building
from sklearn.linear_model import LinearRegression

regresser1 = LinearRegression()
regresser1.fit(x_train, y_train)

y_pred1 = regresser1.predict(x_test)
y_pred1
