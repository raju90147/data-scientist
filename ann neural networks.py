# -*- coding: ut    f-8 -*-
"""
Created on Tue Mar  8 15:51:55 2022

Name: _BOTTA RAJU____________ Batch ID: _05102021__________
Topic: Artificial Neural Networks

"""

# 1. Objective: Build an ANN model to predict the profit of a new startup based on certain features.  
# Import necessary libraries for MLP and reshaping the data structres
import numpy as np
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout,Flatten
from keras.utils import np_utils

np.random.seed(10)

# Loading the data set using pandas as data frame format 
import pandas as pd
startups = pd.read_csv(r"D:\datasets\50_startups.csv")
startups.head(-5)
startups.size
startups.columns

# drop
startups.drop(columns='State', axis=1, inplace=True)


X = startups.drop('Profit', axis = 1)
y = pd.DataFrame(startups['Profit'])

from sklearn.model_selection import train_test_split

X_train ,X_test ,y_train , y_test = train_test_split(X,y , test_size=0.2 , random_state=123)

# storing the no.of classes into the variable nunber of classes
num_of_classes = y_train.shape[1]
num_of_classes
X_train.shape

# creating a user defined function to return the model for which we are 
# giving the input to train the ANN mode
def design_mlp():
  # initializing the model
  model = Sequential()
  model.add(Dense(200, input_dim = 3, activation='relu'))
  model.add(Dense(250, activation='tanh'))
  model.add(Dense(150, activation='tanh'))
  model.add(Dense(300, activation='tanh')) 
  model.add(Dense(num_of_classes, activation='softmax'))
  model.compile(loss="mean_squared_error", optimizer='adam', metrics=['accuracy'])
  return model

# building a CNN Model using train data set and validating on test data set

model = design_mlp()
model

# fitting model on train data
model.fit(x=X_train, y=y_train, batch_size=150, epochs=20)

X_train.shape

# evaluating the model on test data

eval_score_test = model.evaluate(X_test,y_test,verbose=1)
print("Accuracy: %.3f%%" %(eval_score_test[1])*100)

# accuracy score on train data
eval_score_train = model.evaluate(X_train,y_train,verbose=0)
print("Accuracy: %.3f%%" %(eval_score_train[1])*100)

# ================================================****************===============================


# 2. Objective: Predict the burnt area of forest fires with the help of an Artificial Neural Network model

np.random.seed(10)

# Loading the data set using pandas as data frame format 
import pandas as pd
fire_forests = pd.read_csv(r"D:\datasets\fireforests.csv")
fire_forests.head(-5)
fire_forests.size
fire_forests.columns

X = fire_forests.drop('area', axis = 1)
y = pd.DataFrame(fire_forests['area'])

from sklearn.model_selection import train_test_split

X_train ,X_test ,y_train , y_test = train_test_split(X,y , test_size=0.2 , random_state=123)

# storing the no.of classes into the variable nunber of classes
num_of_classes = y_train.shape[1]
num_of_classes
X_train.shape

# creating a user defined function to return the model for which we are 
# giving the input to train the ANN mode
def design_mlp():
  # initializing the model
  model = Sequential()
  model.add(Dense(200, input_dim = 29, activation='relu'))
  model.add(Dense(250, activation='tanh'))
  model.add(Dense(150, activation='tanh'))
  model.add(Dense(300, activation='tanh')) 
  model.add(Dense(num_of_classes, activation='softmax'))
  model.compile(loss="mean_squared_error", optimizer='adam', metrics=['accuracy'])
  return model

# building a CNN Model using train data set and validating on test data set

model = design_mlp()
model

# fitting model on train data
model.fit(x=X_train, y=y_train, batch_size=150, epochs=20)

X_train.shape

# evaluating the model on test data

eval_score_test = model.evaluate(X_test,y_test,verbose=1)
print("Accuracy: %.3f%%" %(eval_score_test[1])*100)

# accuracy score on train data
eval_score_train = model.evaluate(X_train,y_train,verbose=0)
print("Accuracy: %.3f%%" %(eval_score_train[1])*100)

# ========================== ****************** ===============================

# 3. Objective: Build a Neural network model to predict the compressive strength.


np.random.seed(10)

# Loading the data set using pandas as data frame format 
import pandas as pd
concrete = pd.read_csv(r"D:\datasets\concrete.csv")
concrete.head(-5)
concrete.size
concrete.columns

X = concrete.drop('strength', axis = 1)
y = pd.DataFrame(concrete['strength'])

from sklearn.model_selection import train_test_split

X_train ,X_test ,y_train , y_test = train_test_split(X,y , test_size=0.2 , random_state=123)

# storing the no.of classes into the variable nunber of classes
num_of_classes = y_train.shape[1]
num_of_classes
X_train.shape

# creating a user defined function to return the model for which we are 
# giving the input to train the ANN mode
def design_mlp():
  # initializing the model
  model = Sequential()
  model.add(Dense(200, input_dim = 29, activation='relu'))
  model.add(Dense(250, activation='tanh'))
  model.add(Dense(150, activation='tanh'))
  model.add(Dense(300, activation='tanh')) 
  model.add(Dense(num_of_classes, activation='softmax'))
  model.compile(loss="mean_squared_error", optimizer='adam', metrics=['accuracy'])
  return model

# building a CNN Model using train data set and validating on test data set

model = design_mlp()
model

# fitting model on train data
model.fit(x=X_train, y=y_train, batch_size=150, epochs=20)

X_train.shape

# evaluating the model on test data

eval_score_test = model.evaluate(X_test,y_test,verbose=1)
print("Accuracy: %.3f%%" %(eval_score_test[1])*100)

# accuracy score on train data
eval_score_train = model.evaluate(X_train,y_train,verbose=0)
print("Accuracy: %.3f%%" %(eval_score_train[1])*100)


# ================= ************************ ================================

# 4. Objective: solve the problem of churn usig ANN with Exited as target variable.

np.random.seed(10)

# Loading the data set using pandas as data frame format 
import pandas as pd
rpl_bank = pd.read_csv(r"D:\datasets\RPL.csv")
rpl_bank.head(-5)
rpl_bank.size
rpl_bank.columns

# drop
rpl_bank.drop(columns=['RowNumber','CustomerId','Surname'], axis=1, inplace=True)

# label encoding train data
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
rpl_bank['Geography'] = le.fit_transform(rpl_bank['Geography']) 
rpl_bank['Gender'] = le.fit_transform(rpl_bank['Gender']) 

X = rpl_bank.drop('Exited', axis = 1)
y = pd.DataFrame(rpl_bank['Exited'])

from sklearn.model_selection import train_test_split

X_train ,X_test ,y_train , y_test = train_test_split(X,y , test_size=0.2 , random_state=123)

# storing the no.of classes into the variable nunber of classes
num_of_classes = y_train.shape[1]
num_of_classes
X_train.shape

# creating a user defined function to return the model for which we are 
# giving the input to train the ANN mode
def design_mlp():
  # initializing the model
  model = Sequential()
  model.add(Dense(200, input_dim = 29, activation='relu'))
  model.add(Dense(250, activation='tanh'))
  model.add(Dense(150, activation='tanh'))
  model.add(Dense(300, activation='tanh')) 
  model.add(Dense(num_of_classes, activation='softmax'))
  model.compile(loss="mean_squared_error", optimizer='adam', metrics=['accuracy'])
  return model

# building a CNN Model using train data set and validating on test data set

model = design_mlp()
model

# fitting model on train data
model.fit(x=X_train, y=y_train, batch_size=150, epochs=20)

X_train.shape

# evaluating the model on test data

eval_score_test = model.evaluate(X_test,y_test,verbose=1)
print("Accuracy: %.3f%%" %(eval_score_test[1])*100)

# accuracy score on train data
eval_score_train = model.evaluate(X_train,y_train,verbose=0)
print("Accuracy: %.3f%%" %(eval_score_train[1])*100)
