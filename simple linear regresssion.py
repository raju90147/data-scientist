# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 15:26:46 2022

@author: LENOVO
"""

import pandas as pd
import numpy as np           # Weight gained (grams)

calories_consumed = pd.read_csv('D:\Data Set\calories_consumed.csv')
calories_consumed.columns

calories_consumed.info

# EDA 

# Graphical Representation

import matplotlib.pyplot as plt  

plt.bar(height=calories_consumed['Calories Consumed'], x=calories_consumed['Weight gained (grams)'])

plt.hist(calories_consumed['Calories Consumed']) # Histogram

plt.boxplot(calories_consumed['Calories Consumed']) # Boxplot

plt.hist(calories_consumed['Weight gained (grams)']) # Histogram

plt.boxplot(calories_consumed['Weight gained (grams)']) # Boxplot

 
# Scatter plot

plt.scatter(x=calories_consumed['Calories Consumed'], y=calories_consumed['Weight gained (grams)'], color='green')


# correlation

np.corrcoef(calories_consumed['Calories Consumed'], calories_consumed['Weight gained (grams)'])


# covariance

cov_output = np.cov(calories_consumed['Calories Consumed'], calories_consumed['Weight gained (grams)'])[0,1]
cov_output

x = calories_consumed['Calories Consumed']
y = calories_consumed['Weight gained (grams)']

# Import Library

import statsmodels.formula.api as smf

# simple linear regression

model = smf.ols('x~y', data=calories_consumed).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(calories_consumed['Calories Consumed']))
pred1

# Regression Line

plt.scatter(calories_consumed['Calories Consumed'], calories_consumed['Weight gained (grams)'], color='green')
plt.plot(calories_consumed['Calories Consumed'],pred1, 'r')
plt.legend(['Predicted Line', 'Observed Data'])
plt.show()

 
# Error Correlation

res1 = calories_consumed['Calories Consumed']-pred1
res_sqr1 = res1*res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

# Model Building on Transformed data

# Log Transformation

plt.scatter(np.log(calories_consumed['Weight gained (grams)']), calories_consumed['Calories Consumed'])
X = calories_consumed['Weight gained (grams)']
Y = calories_consumed['Calories Consumed']

 
# correlation
np.corrcoef(np.log(calories_consumed['Weight gained (grams)']), calories_consumed['Calories Consumed'])

model2 = smf.ols('np.log(X)~Y', data=calories_consumed).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(calories_consumed['Calories Consumed']))
pred2


# Regression Line

plt.scatter(calories_consumed['Calories Consumed'], calories_consumed['Weight gained (grams)'], color='green')
plt.plot(calories_consumed['Calories Consumed'],pred2, 'r')
plt.legend(['Predicted Line', 'Observed Data'])
plt.show()

 
# Error calculation

res2 = calories_consumed['Calories Consumed']-pred2
res_sqr2 = res2*res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

# Exponential Transformation

plt.scatter(calories_consumed['Weight gained (grams)'], np.log(calories_consumed['Calories Consumed']), color='green')
x1 = calories_consumed['Weight gained (grams)']
y1 = calories_consumed['Calories Consumed']

 
# correlation

np.corrcoef(calories_consumed['Weight gained (grams)']), np.log(calories_consumed['Calories Consumed'])

model2 = smf.ols('x1~np.log(y1)', data=calories_consumed).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(calories_consumed['Calories Consumed']))
pred2

# Regression Line

plt.scatter(calories_consumed['Calories Consumed'], calories_consumed['Weight gained (grams)'], color='yellow')
plt.plot(calories_consumed['Calories Consumed'],pred2, 'r')
plt.legend(['Predicted Line', 'Observed Data'])
plt.show()

 
# Error calculation

res2 = calories_consumed['Calories Consumed']-pred2
res_sqr2 = res2*res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2
 
# -============================== ***************************=================

# 2. Delivery time
# A logistics company recorded the time taken for delivery and the time taken for the sorting of the items for delivery. Build a Simple Linear Regression model to find the relationship between delivery time and sorting time with delivery time as the target variable. Apply necessary transformations and record the RMSE and correlation coefficient values for different models.
import numpy as np
import pandas as pd
delivary = pd.read_csv('D:\Data Set\delivery_time.csv')
delivary.columns

delivary.isna().sum() # checking null values 


# Graphical Representation

import matplotlib.pyplot as plt  

plt.bar(height=delivary['Delivery Time'], x=delivary['Sorting Time'])
plt.hist(delivary['Delivery Time']) # Histogram
plt.boxplot(delivary['Delivery Time']) # Boxplot

plt.hist(delivary['Sorting Time']) # Histogram
plt.boxplot(delivary['Sorting Time']) # Boxplot

# Scatter plot

plt.scatter(x=delivary['Sorting Time'], y=delivary['Delivery Time'], color='green')

# correlation

np.corrcoef(delivary['Delivery Time'], delivary['Sorting Time'])

# covariance

cov_output = np.cov(delivary['Delivery Time'], delivary['Sorting Time'])[0,1]
cov_output

x = delivary['Delivery Time']
y = delivary['Sorting Time']

# Import Library

import statsmodels.formula.api as smf

# simple linear regression

model = smf.ols('x~y', data=delivary).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(delivary['Delivery Time']))
pred1

# Regression Line

plt.scatter(delivary['Delivery Time'], delivary['Sorting Time'], color='green')
plt.plot(delivary['Delivery Time'], pred1, 'r')
plt.legend(['Predicted Line', 'Observed Data'])
plt.show()

# Error Correlation

res1 = delivary['Delivery Time']-pred1 #actual - predicted
res_sqr1 = res1*res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

# Model Building on Transformed data
delivary.columns
# Log Transformation

plt.scatter(np.log(delivary['Sorting Time']), delivary['Delivery Time'])
X = delivary['Sorting Time']
Y = delivary['Delivery Time']
# correlation
np.corrcoef(np.log(delivary['Sorting Time']), delivary['Delivery Time'])

model2 = smf.ols('np.log(X)~Y', data=delivary).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(delivary['Delivery Time']))
pred2

# Regression Line

plt.scatter(delivary['Delivery Time'], delivary['Sorting Time'], color='yellow')
plt.plot(delivary['Delivery Time'],pred2, 'r')
plt.legend(['Predicted Line', 'Observed Data'])
plt.show()

# Error calculation

res2 = delivary['Delivery Time']-pred2
res_sqr2 = res2*res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

# Exponential Transformation

plt.scatter(delivary['Sorting Time'], np.log(delivary['Delivery Time']), color='green')
x1 = delivary['Sorting Time']
y1 = delivary['Delivery Time']

# correlation
np.corrcoef(delivary['Sorting Time']), np.log(delivary['Delivery Time'])

model2 = smf.ols('x1~np.log(y1)', data=delivary).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(delivary['Delivery Time']))
pred2

# Regression Line

plt.scatter(delivary['Delivery Time'], delivary['Sorting Time'], color='purple')
plt.plot(delivary['Delivery Time'],pred2, 'r')
plt.legend(['Predicted Line', 'Observed Data'])
plt.show()

# Error calculation

res2 = calories_consumed['Weight gained (grams)']-pred2
res_sqr2 = res2*res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

# ============================= *************************

# 3. employ data

employee_data = pd.read_csv('D:\Data Set\emp_data.csv')

employee_data.columns

employee_data.isna().sum() # checking null values 

# Graphical Representation

import matplotlib.pyplot as plt  

plt.bar(height=delivary['Delivery Time'], x=delivary['Sorting Time'])
plt.hist(delivary['Delivery Time']) # Histogram
plt.boxplot(delivary['Delivery Time']) # Boxplot

plt.hist(delivary['Sorting Time']) # Histogram
plt.boxplot(delivary['Sorting Time']) # Boxplot

# Scatter plot

plt.scatter(x=employee_data['Salary_hike'], y=employee_data['Churn_out_rate'], color='green')

# correlation

np.corrcoef(employee_data['Salary_hike'], employee_data['Churn_out_rate'])



# covariance

cov_output = np.cov(employee_data['Salary_hike'], employee_data['Churn_out_rate'])[0,1]
cov_output

x = employee_data['Salary_hike']
y = employee_data['Churn_out_rate']

# Import Library

import statsmodels.formula.api as smf

# simple linear regression

model = smf.ols('x~y', data=employee_data).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(employee_data['Salary_hike']))
pred1

# Regression Line

plt.scatter(employee_data['Salary_hike'], employee_data['Churn_out_rate'], color='green')
plt.plot(employee_data['Churn_out_rate'], pred1, 'r')
plt.legend(['Predicted Line', 'Observed Data'])
plt.show()

# Error Correlation

res1 = employee_data['Churn_out_rate']-pred1 #actual - predicted
res_sqr1 = res1*res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

# Model Building on Transformed data
# Log Transformation

plt.scatter(np.log(employee_data['Salary_hike']), employee_data['Churn_out_rate'])
X = employee_data['Salary_hike']
Y = employee_data['Churn_out_rate']

# correlation

np.corrcoef(np.log(employee_data['Salary_hike']), employee_data['Churn_out_rate'])

model2 = smf.ols('np.log(X)~Y', data=employee_data).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(employee_data['Churn_out_rate']))
pred2

# Regression Line

plt.scatter(employee_data['Churn_out_rate'], employee_data['Salary_hike'], color='yellow')
plt.plot(employee_data['Churn_out_rate'],pred2, 'r')
plt.legend(['Predicted Line', 'Observed Data'])
plt.show()

# Error calculation

res2 = employee_data['Churn_out_rate']-pred2
res_sqr2 = res2*res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

# Exponential Transformation

plt.scatter(employee_data['Salary_hike'], np.log(employee_data['Churn_out_rate']), color='green')
x1 = employee_data['Salary_hike']
y1 = employee_data['Churn_out_rate']

# correlation

np.corrcoef(employee_data['Salary_hike']), np.log(employee_data['Churn_out_rate'])

model2 = smf.ols('x1~np.log(y1)', data=employee_data).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(employee_data['Churn_out_rate']))
pred2

# Regression Line

plt.scatter(employee_data['Churn_out_rate'], employee_data['Salary_hike'], color='purple')
plt.plot(employee_data['Churn_out_rate'], pred2, 'r')
plt.legend(['Predicted Line', 'Observed Data'])
plt.show()

# Error calculation

res2 = employee_data['Salary_hike']-pred2
res_sqr2 = res2*res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

# =================== ************************ ============================

# 4. Salary_Data

import pandas as pd
import numpy as np
salary_data = pd.read_csv('D:\Data Set\Salary_Data.csv')
salary_data.columns

salary_data.isna().sum() # checking null values 

# Graphical Representation

import matplotlib.pyplot as plt  

plt.bar(height=salary_data['Salary'], x=salary_data['YearsExperience'])
plt.hist(salary_data['Salary']) # Histogram
plt.boxplot(salary_data['Salary']) # Boxplot

plt.hist(salary_data['YearsExperience']) # Histogram
plt.boxplot(salary_data['YearsExperience']) # Boxplot

# Scatter plot

plt.scatter(x=salary_data['Salary'], y=salary_data['YearsExperience'], color='green')

# correlation

np.corrcoef(salary_data['Salary'], salary_data['YearsExperience'])

# covariance

cov_output = np.cov(salary_data['Salary'], salary_data['YearsExperience'])[0,1]
cov_output

x = salary_data['Salary']
y = salary_data['YearsExperience']

# Import Library

import statsmodels.formula.api as smf

# simple linear regression

model = smf.ols('x~y', data=salary_data).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(salary_data['Salary']))
pred1

# Regression Line

plt.scatter(salary_data['Salary'], salary_data['YearsExperience'], color='green')
plt.plot(salary_data['Salary'], pred1, 'r')
plt.legend(['Predicted Line', 'Observed Data'])
plt.show()

# Error Correlation

res1 = salary_data['Salary']-pred1 #actual - predicted
res_sqr1 = res1*res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

# Model Building on Transformed data
# Log Transformation

plt.scatter(np.log(salary_data['Salary']), salary_data['YearsExperience'])
X = salary_data['YearsExperience']
Y = salary_data['Salary']
# correlation
np.corrcoef(np.log(salary_data['Salary']), salary_data['YearsExperience'])

model2 = smf.ols('np.log(X)~Y', data=salary_data).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(salary_data['YearsExperience']))
pred2 

# Regression Line

plt.scatter(salary_data['Salary'], salary_data['YearsExperience'], color='yellow')
plt.plot(salary_data['Salary'],pred2, 'r')
plt.legend(['Predicted Line', 'Observed Data'])
plt.show()

# Error calculation

res2 = salary_data['Salary']-pred2
res_sqr2 = res2*res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

# Exponential Transformation

plt.scatter(salary_data['YearsExperience'], np.log(salary_data['Salary']), color='green')
x1 = salary_data['YearsExperience']
y1 = salary_data['Salary']

# correlation
np.corrcoef(salary_data['YearsExperience']), np.log(salary_data['Salary'])

model2 = smf.ols('x1~np.log(y1)', data=salary_data).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(salary_data['Salary']))
pred2

# Regression Line

plt.scatter(salary_data['Salary'], salary_data['YearsExperience'], color='purple')
plt.plot(salary_data['Salary'], pred2, 'r')
plt.legend(['Predicted Line', 'Observed Data'])
plt.show()

# Error calculation

res2 = employee_data['Salary_hike']-pred2
res_sqr2 = res2*res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2
    
# ========================== ************************ ===================

# 5. SAT Scores

gpa = pd.read_csv('D:\Data Set\SAT_GPA.csv')
gpa.columns
gpa.isna().sum() # checking null values 

# Graphical Representation

import matplotlib.pyplot as plt  

plt.bar(height=gpa['GPA'], x=gpa['SAT_Scores'])
plt.hist(gpa['GPA']) # Histogram
plt.boxplot(gpa['SAT_Scores']) # Boxplot

plt.hist(gpa['SAT_Scores']) # Histogram
plt.boxplot(gpa['SAT_Scores']) # Boxplot

# Scatter plot

plt.scatter(x=gpa['SAT_Scores'], y=gpa['GPA'], color='green')

# correlation

np.corrcoef(gpa['SAT_Scores'], gpa['GPA'])

# covariance

cov_output = np.cov(gpa['GPA'], gpa['SAT_Scores'])[0,1]
cov_output

x = gpa['SAT_Scores']
y = gpa['GPA']

# Import Library

import statsmodels.formula.api as smf

# simple linear regression

model = smf.ols('x~y', data=gpa).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(gpa['GPA']))
pred1

# Regression Line

plt.scatter(gpa['GPA'], gpa['SAT_Scores'], color='green')
plt.plot(gpa['GPA'], pred1, 'r')
plt.legend(['Predicted Line', 'Observed Data'])
plt.show()

# Error Correlation

res1 = gpa['GPA']-pred1 #actual - predicted
res_sqr1 = res1*res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

# Model Building on Transformed data
# Log Transformation

plt.scatter(np.log(gpa['GPA']), gpa['SAT_Scores'])
X = gpa['SAT_Scores']
Y = gpa['GPA']
# correlation
np.corrcoef(np.log(gpa['GPA']), gpa['SAT_Scores'])

model2 = smf.ols('np.log(X)~Y', data=gpa).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(gpa['SAT_Scores']))
pred2 

# Regression Line

plt.scatter(gpa['GPA'], gpa['SAT_Scores'], color='yellow')
plt.plot(gpa['GPA'],pred2, 'r')
plt.legend(['Predicted Line', 'Observed Data'])
plt.show()

# Error calculation

res2 = salary_data['Salary']-pred2
res_sqr2 = res2*res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

# Exponential Transformation

plt.scatter(gpa['SAT_Scores'], np.log(gpa['GPA']), color='green')
x1 = gpa['SAT_Scores']
y1 = gpa['GPA']

# correlation
np.corrcoef(gpa['SAT_Scores']), np.log(gpa['GPA'])

model2 = smf.ols('x1~np.log(y1)', data=gpa).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(gpa['GPA']))
pred2

# Regression Line

plt.scatter(gpa['GPA'], gpa['SAT_Scores'], color='purple')
plt.plot(gpa['GPA'], pred2, 'r')
plt.legend(['Predicted Line', 'Observed Data'])
plt.show()

# Error calculation

res2 = gpa['SAT_Scores']-pred2
res_sqr2 = res2*res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


