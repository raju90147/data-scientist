# -*- coding: utf-8 -*-
"""
Name:	 BOTTA RAJU
Batch ID: 05102021		
Topic: Forecasting – Time Series

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # Holt Winter's Exponential Smoothing

airlines = pd.read_excel("D:\datasets/Airlines Data.xlsx")
airlines.head()

# time series plot
airlines.Passengers.plot()

# splitting the data into train and test data

train = airlines.head(77)
test = airlines.tail(20)

# creating a function to calculate the MAPE value for test data

def MAPE(pred, air):
  temp = np.abs((pred-air)/air)*100
  return np.mean(temp)

# Moving Average for the time series
mv_pred = airlines["Passengers"].rolling(5).mean()
mv_pred.tail(20)
MAPE(mv_pred.tail(20), test.Passengers)


# Plot with Moving Averages
airlines.Passengers.plot(label = "air")
for i in range(2, 12, 2):
    airlines["Passengers"].rolling(i).mean().plot(label = str(i))
plt.legend(loc = 3)

    
# Time series decomposition is the process of separating data into its core components.
# Time series decomposition plot using Moving Average
decompose_ts_add = seasonal_decompose(airlines.Passengers, model = "additive", period = 12)
print(decompose_ts_add.trend)
print(decompose_ts_add.seasonal)
print(decompose_ts_add.resid)
print(decompose_ts_add.observed)
decompose_ts_add.plot()

decompose_ts_mul = seasonal_decompose(airlines.Passengers, model = "multiplicative", period = 4)
decompose_ts_mul.plot()

# ACF and PACF plot on Original data sets 
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(airlines.Passengers, lags = 4)
tsa_plots.plot_pacf(airlines.Passengers, lags=4)
# ACF is an (complete) auto-correlation function gives values 
# of auto-correlation of any time series with its lagged values.
# PACF is a partial auto-correlation function. 
# It finds correlations of present with lags of the residuals of the time series

# Simple Exponential Method
ses_model = SimpleExpSmoothing(train["Passengers"]).fit()
pred_ses = ses_model.predict(start = test.index[0], end = test.index[-1])
MAPE(pred_ses, test.Passengers) 


# Holt method 
hw_model = Holt(train["Passengers"]).fit()
pred_hw = hw_model.predict(start = test.index[0], end = test.index[-1])
MAPE(pred_hw, test.Passengers) 


# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(train["Passengers"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = test.index[0], end = test.index[-1])
MAPE(pred_hwe_add_add, test.Passengers) 

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(train["Passengers"], seasonal = "mul", trend = "add", seasonal_periods = 4).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = test.index[0], end = test.index[-1])
MAPE(pred_hwe_mul_add, test.Passengers) 

# Final Model on 100% Data
hwe_model_add_add = ExponentialSmoothing(airlines["Passengers"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()

# Load the new data which includes the entry for future 4 values
new_data = pd.read_excel("D:\datasets/new airlines data.xlsx")

newdata_pred = hwe_model_add_add.predict(start = new_data.index[0], end = new_data.index[-1])
newdata_pred

# =================================================== **************** ===================

# 2. Objective: Predict sales for the next two years by using time series forecasting

cola_sales = pd.read_excel("D:\datasets/Cocacola_Sales_Rawdata.xlsx")
cola_sales.head()

# time series plot
cola_sales.Sales.plot()

# splitting the data into train and test data

train = cola_sales.head(34)
test = cola_sales.tail(8)

# creating a function to calculate the MAPE value for test data

def MAPE(pred, coke):
  temp = np.abs((pred-coke)/coke)*100
  return np.mean(temp)

# Moving Average for the time series
mv_pred = cola_sales["Sales"].rolling(4).mean()
mv_pred.tail(20)
MAPE(mv_pred.tail(20), test.Sales)


# Plot with Moving Averages
cola_sales.Sales.plot(label = "coke")
for i in range(2, 9, 2):
    cola_sales["Sales"].rolling(i).mean().plot(label = str(i))
plt.legend(loc = 3)
    
# Time series decomposition is the process of separating data into its core components.
# Time series decomposition plot using Moving Average
decompose_ts_add = seasonal_decompose(cola_sales.Sales, model = "additive", period = 4)
print(decompose_ts_add.trend)
print(decompose_ts_add.seasonal)
print(decompose_ts_add.resid)
print(decompose_ts_add.observed)
decompose_ts_add.plot()

decompose_ts_mul = seasonal_decompose(cola_sales.Sales, model = "multiplicative", period = 4)
decompose_ts_mul.plot()

# ACF and PACF plot on Original data sets 
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(cola_sales.Sales, lags = 4)
tsa_plots.plot_pacf(cola_sales.Sales, lags=4)
# ACF is an (complete) auto-correlation function gives values 
# of auto-correlation of any time series with its lagged values.
# PACF is a partial auto-correlation function. 
# It finds correlations of present with lags of the residuals of the time series

# Simple Exponential Method
ses_model = SimpleExpSmoothing(train["Sales"]).fit()
pred_ses = ses_model.predict(start = test.index[0], end = test.index[-1])
MAPE(pred_ses, test.Sales) 

# Holt method 
hw_model = Holt(train["Sales"]).fit()
pred_hw = hw_model.predict(start = test.index[0], end = test.index[-1])
MAPE(pred_hw, test.Sales) 

# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(train["Sales"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = test.index[0], end = test.index[-1])
MAPE(pred_hwe_add_add, test.Sales) 

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(train["Sales"], seasonal = "mul", trend = "add", seasonal_periods = 4).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = test.index[0], end = test.index[-1])
MAPE(pred_hwe_mul_add, test.Sales) 

# Final Model on 100% Data
hwe_model_add_add = ExponentialSmoothing(cola_sales["Sales"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()

# Load the new data which includes the entry for future 4 values
new_data = pd.read_excel("D:\datasets/Newdata_CocaCola_Sales.xlsx")

newdata_pred = hwe_model_add_add.predict(start = new_data.index[0], end = new_data.index[-1])
newdata_pred


# ================================= ****************** ======================

#  3.. Objective:  To forecast the sale for the next year

plastic_sales = pd.read_csv("D:\datasets/PlasticSales.csv")
plastic_sales.head()

# time series plot
plastic_sales.Sales.plot()

# splitting the data into train and test data

train = plastic_sales.head(34)
test = plastic_sales.tail(8)

# creating a function to calculate the MAPE value for test data

def MAPE(pred, plastic):
  temp = np.abs((pred-plastic)/plastic)*100
  return np.mean(temp)

# Moving Average for the time series
mv_pred = plastic_sales["Sales"].rolling(4).mean()
mv_pred.tail(20)
MAPE(mv_pred.tail(20), test.Sales)

# Plot with Moving Averages
plastic_sales.Sales.plot(label = "coke")
for i in range(2, 9, 2):
    plastic_sales["Sales"].rolling(i).mean().plot(label = str(i))
plt.legend(loc = 3)
    
# Time series decomposition is the process of separating data into its core components.
# Time series decomposition plot using Moving Average
decompose_ts_add = seasonal_decompose(plastic_sales.Sales, model = "additive", period = 12)
print(decompose_ts_add.trend)
print(decompose_ts_add.seasonal)
print(decompose_ts_add.resid)
print(decompose_ts_add.observed)
decompose_ts_add.plot()

decompose_ts_mul = seasonal_decompose(plastic_sales.Sales, model = "multiplicative", period = 12)
decompose_ts_mul.plot()

# ACF and PACF plot on Original data sets 
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(plastic_sales.Sales, lags = 4)
tsa_plots.plot_pacf(plastic_sales.Sales, lags=4)
# ACF is an (complete) auto-correlation function gives values 
# of auto-correlation of any time series with its lagged values.
# PACF is a partial auto-correlation function. 
# It finds correlations of present with lags of the residuals of the time series

# Simple Exponential Method
ses_model = SimpleExpSmoothing(train["Sales"]).fit()
pred_ses = ses_model.predict(start = test.index[0], end = test.index[-1])
MAPE(pred_ses, test.Sales) 

# Holt method 
hw_model = Holt(train["Sales"]).fit()
pred_hw = hw_model.predict(start = test.index[0], end = test.index[-1])
MAPE(pred_hw, test.Sales) 

# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(train["Sales"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = test.index[0], end = test.index[-1])
MAPE(pred_hwe_add_add, test.Sales) 

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(train["Sales"], seasonal = "mul", trend = "add", seasonal_periods = 4).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = test.index[0], end = test.index[-1])
MAPE(pred_hwe_mul_add, test.Sales) 

# Final Model on 100% Data
hwe_model_add_add = ExponentialSmoothing(plastic_sales["Sales"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()

# Load the new data which includes the entry for future 4 values
new_data = pd.read_excel("D:\datasets/Newdata_CocaCola_Sales.xlsx")

newdata_pred = hwe_model_add_add.predict(start = new_data.index[0], end = new_data.index[-1])
newdata_pred

# =================== ************************* =======================

# 4. Object: build a forecasting model on the data above for solar power consumption.

solarpower = pd.read_csv("D:\datasets/solarpower_cumuldaybyday2.csv")
solarpower.head()

# time series plot
solarpower.cum_power.plot()

# splitting the data into train and test data
train = solarpower.head(2047)
test = solarpower.tail(511)

# creating a function to calculate the MAPE value for test data

def MAPE(pred, solar):
  temp = np.abs((pred-solar)/solar)*100
  return np.mean(temp)

# Moving Average for the time series
mv_pred = solarpower["cum_power"].rolling(4).mean()
mv_pred.tail(20)
MAPE(mv_pred.tail(20), test.Sales)

# Plot with Moving Averages
solarpower.cum_power.plot(label = "solar")
for i in range(2, 9, 2):
    solarpower["cum_power"].rolling(i).mean().plot(label = str(i))
plt.legend(loc = 3)
    
# Time series decomposition is the process of separating data into its core components.
# Time series decomposition plot using Moving Average
decompose_ts_add = seasonal_decompose(solarpower.cum_power, model = "additive", period = 12)
print(decompose_ts_add.trend)
print(decompose_ts_add.seasonal)
print(decompose_ts_add.resid)
print(decompose_ts_add.observed)
decompose_ts_add.plot()

decompose_ts_mul = seasonal_decompose(solarpower.Sales, model = "multiplicative", period = 12)
decompose_ts_mul.plot()

# ACF and PACF plot on Original data sets 
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(solarpower.cum_power, lags = 4)
tsa_plots.plot_pacf(solarpower.cum_power, lags=4)
# ACF is an (complete) auto-correlation function gives values 
# of auto-correlation of any time series with its lagged values.
# PACF is a partial auto-correlation function. 
# It finds correlations of present with lags of the residuals of the time series

# Simple Exponential Method
ses_model = SimpleExpSmoothing(train["cum_power"]).fit()
pred_ses = ses_model.predict(start = test.index[0], end = test.index[-1])
MAPE(pred_ses, test.cum_power) 

# Holt method 
hw_model = Holt(train["cum_power"]).fit()
pred_hw = hw_model.predict(start = test.index[0], end = test.index[-1])
MAPE(pred_hw, test.cum_power) 

# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(train["cum_power"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = test.index[0], end = test.index[-1])
MAPE(pred_hwe_add_add, test.cum_power) 

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(train["cum_power"], seasonal = "mul", trend = "add", seasonal_periods = 4).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = test.index[0], end = test.index[-1])
MAPE(pred_hwe_mul_add, test.cum_power) 

# Final Model on 100% Data
hwe_model_add_add = ExponentialSmoothing(solarpower["cum_power"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()
