# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 15:10:59 2022
Name: __BOTTA RAJU___________ Batch ID: 05102021___________
Topic: Lasso and Ridge Regression

"""
#1. objective: to improve the customer experience by providing them online predictive prices for their laptops

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns


# loading the data
laptop = pd.read_csv("/content/Computer_Data (1).csv")

laptop.columns
laptop.head()

# dropping columns

laptop.drop(columns=['Unnamed: 0'], axis=1, inplace=True)

# Correlation matrix 
a = laptop.corr()
a

# EDA
a1 = laptop.describe()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
laptop['cd'] = le.fit_transform(laptop['cd']) 
laptop['multi'] = le.fit_transform(laptop['multi']) 
laptop['premium'] = le.fit_transform(laptop['premium']) 

# Sctter plot to know relation between variables
sns.pairplot(laptop) 

# Preparing the model on train data 
model_train = smf.ols("price ~ speed + hd + ram + screen + ads + trend", data = laptop).fit()
model_train.summary()


# Prediction
pred = model_train.predict(laptop)

# Error
resid  = pred - laptop.price

# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse


# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###
from sklearn.linear_model import Lasso
help(Lasso)

lasso = Lasso(alpha = 0.13, normalize = True)

lasso.fit(laptop.iloc[:, 1:], laptop.price)

# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(laptop.columns[1:]))

lasso.alpha

pred_lasso = lasso.predict(laptop.iloc[:, 1:])


# Adjusted r-square
lasso.score(laptop.iloc[:, 1:], laptop.price)

# RMSE
np.sqrt(np.mean((pred_lasso - laptop.price)**2))

### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
help(Ridge)
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(laptop.iloc[:, 1:], laptop.price)


# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(laptop.columns[1:]))

rm.alpha

pred_rm = rm.predict(laptop.iloc[:, 1:])


# Adjusted r-square
rm.score(laptop.iloc[:, 1:], laptop.price)

# RMSE
np.sqrt(np.mean((pred_rm - laptop.price)**2))


### ELASTIC NET REGRESSION ###
from sklearn.linear_model import ElasticNet 
help(ElasticNet)
enet = ElasticNet(alpha = 0.4)

enet.fit(laptop.iloc[:, 1:], laptop.price) 


# Coefficients values for all the independent vairbales
enet.coef_
enet.intercept_

plt.bar(height = pd.Series(enet.coef_), x = pd.Series(laptop.columns[1:]))

enet.alpha

pred_enet = enet.predict(laptop.iloc[:, 1:])

# Adjusted r-square
enet.score(laptop.iloc[:, 1:], laptop.price)

# RMSE
np.sqrt(np.mean((pred_enet - laptop.price)**2))


# Lasso Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}


lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(laptop.iloc[:, 1:], laptop.price)


lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred = lasso_reg.predict(laptop.iloc[:, 1:])


# Adjusted r-square#
lasso_reg.score(laptop.iloc[:, 1:], laptop.price)

# RMSE
np.sqrt(np.mean((lasso_pred - laptop.price)**2))


# Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge


ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(laptop.iloc[:, 1:], laptop.price)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(laptop.iloc[:, 1:])


# Adjusted r-square#
ridge_reg.score(laptop.iloc[:, 1:], laptop.price)

# RMSE
np.sqrt(np.mean((ridge_pred - laptop.price)**2))

# ElasticNet Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

enet = ElasticNet()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 5)
enet_reg.fit(laptop.iloc[:, 1:], laptop.price)

enet_reg.best_params_
enet_reg.best_score_

enet_pred = enet_reg.predict(laptop.iloc[:, 1:])


# Adjusted r-square
enet_reg.score(laptop.iloc[:, 1:], laptop.price)

# RMSE
np.sqrt(np.mean((enet_pred - laptop.price)**2))

# ====================== ****************** ==========================

#2. Objective: predict the price of the car once the user inputs the required factors

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# loading the data
car_sales = pd.read_csv('/content/ToyotaCorolla (1).csv', encoding='unicode_escape')
car_sales.columns

# dropping columns

car_sales.drop(columns=['Id','Model','Power_Steering','Mfg_Month','Mfg_Year','Fuel_Type','Met_Color','Color','Automatic','Cylinders','Mfr_Guarantee','BOVAG_Guarantee','Guarantee_Period','ABS','Airbag_1','Airbag_2','Airco','Automatic_airco','Boardcomputer','CD_Player','Central_Lock','Powered_Windows','Radio','Mistlamps','Sport_Model','Backseat_Divider','Metallic_Rim','Radio_cassette','Tow_Bar'], axis=1, inplace=True)
car_sales.head()

# pair plots
sns.pairplot(car_sales)

# preparing the model on train data

model_train = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight', data=car_sales).fit()
model_train.summary()

# Prediction

pred = model_train.predict(car_sales)

# ERROR
resid  = pred-car_sales.Price

# RMSE values for data

rmse = np.sqrt(np.mean(resid*resid))
rmse

# coefficient values for all independent variable 

lasso.coef_
lasso.intercept_

plt.bar(height=pd.Series(lasso.coef_), x=pd.Series(car_sales.columns[0:]))

lasso.alpha

pred_lasso = lasso.predict(car_sales.iloc[:,0:])

# Adjusted R-Square

lasso.score(car_sales.iloc[:,0:], car_sales.Price)

# RMSE

np.sqrt(np.mean(pred_lasso)**2)

# Ridge Regression

### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
help(Ridge)
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(car_sales.iloc[:, 1:], car_sales.Price)

# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(car_sales.columns[1:]))

rm.alpha

pred_rm = rm.predict(car_sales.iloc[:, 1:])

# Adjusted r-square
rm.score(car_sales.iloc[:, 1:], car_sales.Price)

# RMSE
np.sqrt(np.mean((pred_rm - car_sales.Price)**2))

### ELASTIC NET REGRESSION ###
from sklearn.linear_model import ElasticNet 
help(ElasticNet)
enet = ElasticNet(alpha = 0.4)

enet.fit(car_sales.iloc[:, 1:], car_sales.Price) 


### ELASTIC NET REGRESSION ###
from sklearn.linear_model import ElasticNet 

enet = ElasticNet()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}


enet.fit(car_sales.iloc[:, 1:], car_sales.Price) 

# Coefficients values for all the independent vairbales
enet.coef_
enet.intercept_

plt.bar(height = pd.Series(enet.coef_), x = pd.Series(car_sales.columns[1:]))

enet.alpha

pred_enet = enet.predict(car_sales.iloc[:, 1:])


# Adjusted r-square
enet.score(car_sales.iloc[:, 1:], car_sales.Price)

# RMSE
np.sqrt(np.mean((pred_enet - car_sales.Price)**2))

# Lasso Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(car_sales.iloc[:, 1:], car_sales.Price)

lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred = lasso_reg.predict(car_sales.iloc[:, 1:])


# Adjusted r-square#
lasso_reg.score(car_sales.iloc[:, 1:], car_sales.Price)

# RMSE
np.sqrt(np.mean((lasso_pred - car_sales.Price)**2))

# Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(car_sales.iloc[:, 1:], car_sales.Price)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(car_sales.iloc[:, 1:])


# Adjusted r-square#
ridge_reg.score(car_sales.iloc[:, 1:], car_sales.Price)

# RMSE
np.square(np.mean(enet_pred)**2)


# ElasticNet Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

enet = ElasticNet()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 5)
enet_reg.fit(car_sales.iloc[:, 1:], car_sales.Price)

enet_reg.best_params_
enet_reg.best_score_

enet_pred = enet_reg.predict(car_sales.iloc[:, 1:])

# Adjusted r-square
enet_reg.score(car_sales.iloc[:, 1:], car_sales.Price)

# RMSE
np.sqrt(np.mean((enet_pred - car_sales.Price)**2))

# ==================================== ************************** ==============

#3. to analyze the data and build a Lasso and Ridge Regression model

# loading the data
life = pd.read_csv("C:/Datasets_BA/Linear Regression/Life_expectancy.csv")

# Rearrange the order of the variables
life = life.iloc[:, [1, 0, 2, 3, 4]]
life.columns

# Correlation matrix 
a = life.corr()
a

# EDA
a1 = life.describe()

life = life.iloc[:, [3, 0, 1, 2, 4, 5, 6 ]]

# label encoding train data

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
life['Country'] = le.fit_transform(life['Country']) 
life['Status'] = le.fit_transform(life['Status']) 



# Sctter plot and histogram between variables
import seaborn as sns
sns.pairplot(life) 

# Preparing the model on train data 
import statsmodels.formula.api as smf

model_train = smf.ols("Life_expectancy ~ Country + Year + Status + Adult_Mortality + infant_deaths + Alcohol + percentage_expenditure + Hepatitis_B + Measles + BMI + under_five_deaths + Polio + Total_expenditure + Diphtheria + HIV_AIDS + GDP + Population + thinness + thinness_yr + Income_composition + Schooling", data = life).fit()
model_train.summary()


# Prediction
pred = model_train.predict(life)
# Error
resid  = pred - life.Life_expectancy

# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse

# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###
from sklearn.linear_model import Lasso
help(Lasso)

lasso = Lasso(alpha = 0.13, normalize = True)

lasso.fit(life.iloc[:, 1:], life.Life_expectancy)

# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(life.columns[1:]))

lasso.alpha

pred_lasso = lasso.predict(life.iloc[:, 1:])

# Adjusted r-square
lasso.score(life.iloc[:, 1:], life.MPG)

# RMSE
np.sqrt(np.mean((pred_lasso - life.Life_expectancy)**2))


### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
help(Ridge)
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(life.iloc[:, 1:], life.Life_expectancy)

# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(life.columns[1:]))

rm.alpha

pred_rm = rm.predict(life.iloc[:, 1:])

# Adjusted r-square
rm.score(life.iloc[:, 1:], life.Life_expectancy)

# RMSE
np.sqrt(np.mean((pred_rm - life.Life_expectancy)**2))


### ELASTIC NET REGRESSION ###
from sklearn.linear_model import ElasticNet 
help(ElasticNet)
enet = ElasticNet(alpha = 0.4)

enet.fit(life.iloc[:, 1:], life.Life_expectancy) 

# Coefficients values for all the independent vairbales
enet.coef_
enet.intercept_
import matplotlib.pyplot as plt

plt.bar(height = pd.Series(enet.coef_), x = pd.Series(life.columns[1:]))

enet.alpha

pred_enet = enet.predict(life.iloc[:, 1:])

# Adjusted r-square
enet.score(life.iloc[:, 1:], life.Life_expectancy)

# RMSE
np.sqrt(np.mean((pred_enet - life.Life_expectancy)**2))


####################

# Lasso Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(life.iloc[:, 1:], life.Life_expectancy)


lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred = lasso_reg.predict(life.iloc[:, 1:])

# Adjusted r-square#
lasso_reg.score(life.iloc[:, 1:], life.Life_expectancy)

# RMSE
np.sqrt(np.mean((lasso_pred - life.Life_expectancy)**2))



# Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(life.iloc[:, 1:], life.MPG)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(life.iloc[:, 1:])

# Adjusted r-square#
ridge_reg.score(life.iloc[:, 1:], life.Life_expectancy)

# RMSE
np.sqrt(np.mean((ridge_pred - life.Life_expectancy)**2))



# ElasticNet Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

enet = ElasticNet()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 5)
enet_reg.fit(life.iloc[:, 1:], life.Life_expectancy)

enet_reg.best_params_
enet_reg.best_score_

enet_pred = enet_reg.predict(life.iloc[:, 1:])

# Adjusted r-square
enet_reg.score(life.iloc[:, 1:], life.Life_expectancy)

# RMSE
np.sqrt(np.mean((enet_pred - life.Life_expectancy)**2))

