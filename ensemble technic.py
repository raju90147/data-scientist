# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:05:47 2022

Name : Botta Raju, Batch ID: 05102021
Topic: Ensemble technics / Bagging classification

"""

# Bagging

import pandas as pd
diabeted = pd.read_csv('D:\Data Set\Diabeted_Ensemble.csv')
diabeted.columns
diabeted.head()
diabeted.info()

# input & output split

predictors = diabeted.loc[:, diabeted.columns!=' Class variable']
type(predictors)
target = diabeted[' Class variable']
type(target)

#Train test partition of the data

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size=0.2, random_state=0)

from sklearn import tree

clftree = tree.DecisionTreeClassifier()

from sklearn.ensemble import BaggingClassifier

bag_clf = BaggingClassifier(base_estimator=clftree, n_estimators=500, bootstrap=True, n_jobs=1, random_state=42)
bag_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing data

confusion_matrix(y_test, bag_clf.predict(x_test))
accuracy_score(y_test, bag_clf.predict(x_test))

#Evaluation on Training data

confusion_matrix(y_train, bag_clf.predict(x_train))
accuracy_score(y_train, bag_clf.predict(x_train))

#Boosting

from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(learning_rate=0.02, n_estimators=500)
ada_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

#Evaluation on testing data

confusion_matrix(y_test, ada_clf.predict(x_test))
accuracy_score(y_test, ada_clf.predict(x_test))

#Evaluation on training data
confusion_matrix(y_train, ada_clf.predict(x_train))
accuracy_score(y_train, ada_clf.predict(x_train))

# Gradient Boosting

from sklearn.ensemble import GradientBoostingClassifier

gb_clf = GradientBoostingClassifier()
gb_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(y_test, gb_clf.predict(x_test))
accuracy_score(y_test, gb_clf.predict(x_test))


#Hyper parameters

glb_clf2 = GradientBoostingClassifier(learning_rate=0.02, n_estimators=10000, max_depth=1)
glb_clf2.fit(x_train, y_train)

# accuracy
from sklearn.metrics import accuracy_score, confusion_matrix

# evaluation on testing data
confusion_matrix(y_test, glb_clf2.predict(x_test))
accuracy_score(y_test, glb_clf2.predict(x_test))

# evaluation on training data

confusion_matrix(y_train, glb_clf2.predict(x_train))
accuracy_score(y_train, glb_clf2.predict(x_train))


# XG BOOST

import xgboost as xgb

xgb_clf = xgb.XGBClassifier(max_depth=5, n_estimators=10000, learning_rate=0.3, n_jobs=-1)

# n_jobs - no.of parallel threads used to run xg boost
# learning_rate(float) - Boosting learning rate (xgb's "eta")

xgb_clf.fit(x_train, y_train)

# evaluation on test data
confusion_matrix(y_test, xgb_clf.predict(x_test))
accuracy_score(y_test, xgb_clf.predict(x_test))

# evaluation on train data
confusion_matrix(y_train, xgb_clf.predict(x_train))
accuracy_score(y_train, xgb_clf.predict(x_train))

xgb.plot_importance(xgb_clf)

xgb_clf = xgb.XGBClassifier(n_estimators=500, learning_rate=0.1, random_state=42)

#n = range(3,10,2)
param_test1 = {'max_depth': range(3,10,2),'gamma':[0.1,0.2,0.3], 'subsample':[0.8,0.9],'colsample_bytree':[0.8,0.9], 'rag_alpha':[1e-2]}

# Grid Search

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(xgb_clf, param_test1, n_jobs=-1, cv=5, scoring='accuracy')

grid_search.fit(x_train, y_train)
cv_xg_clf = grid_search.best_estimator_

#Evaluate on Testing data with model with hyper parameter

accuracy_score(y_test, cv_xg_clf.predict(x_test))


# Stacking

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
# from sklearn SVC import LinearSVC
from mlxtend.classifier import StackingClassifier
import warnings
warnings.filterwarnings('ignore')

# Define base learners

myclf1 = KNeighborsClassifier(n_neighbors=1)
myclf2 = DecisionTreeClassifier(max_depth=4, random_state=123456)
myclf3 = MLPClassifier(hidden_layer_sizes =(100, ), solver='lbfgs', random_state=123456)

# Defining meta model

mylr = LogisticRegression()

# Creating stacking classifier with above models


stackingclf = StackingClassifier(classifiers=[myclf1,myclf2,myclf3], meta_classifier=mylr)

# kf = model_selection.KFold()

print('Doing 3 fold cross validation here:\n')

for iterclf, iterabel in zip([myclf1,myclf2,myclf3,stackingclf], ['K-Nearest Neighbors Model','Decision Tree Classifier','MLP Classifier']):
    
    scores = model_selection.cross_val_score(iterclf, x_train, y_train, cv=3, scoring='accuracy')
    print('Accuracy: %0.3f(+/-%0.3f) [%s]' %(scores.mean(), scores.std(), iterabel))




#  Voting (Hard & soft)

from sklearn.ensemble import VotingClassifier
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Instatiate the learners (Classifiers)

learner_1 = KNeighborsClassifier(n_neighbors=5)
learner_2 = Perceptron(tol=1e-2, random_state=0)
learner_3 = SVC(gamma=0.001)

# Instantiate the voting classifier

voting = VotingClassifier([('KNN',learner_1),
                           ('Prc',learner_2),
                           ('svm', learner_3)])
 
# Fit Classifier with training data

voting.fit(x_train, y_train)

#Predict the most voted class

hard_predictions = voting.predict(x_test)

# Accuracy of hard voting

print('Hard Voting:', accuracy_score(y_test, hard_predictions))


# SOFT VOTING

# Instatiate the learners (classifier)                         

learner_4 = DecisionTreeClassifier(max_depth=4, random_state=123456)
learner_5 = GaussianNB()
learner_6 = SVC(gamma=0.001, probability=True)

# Instantiating the voting classifier

voting = VotingClassifier([('KNN',learner_4),
                          ('NB',learner_5),
                          ('SVM', learner_6)], voting= 'soft')

# Fit classifier with the training data

voting.fit(x_train, y_train)
learner_4.fit(x_train, y_train)
learner_5.fit(x_train, y_train)
learner_6.fit(x_train, y_train)

# Predict the most probable class

soft_predictions = voting.predict(x_test)

# Get the base learner predictions

prediction_4 = learner_4.predict(x_test)
prediction_5 = learner_5.predict(x_test)
prediction_6 = learner_6.predict(x_test)

# Accuracies of base learners

print('L4:', accuracy_score(y_test, prediction_4))
print('L5:', accuracy_score(y_test, prediction_5))
print('L6:', accuracy_score(y_test, prediction_6))

# Accuracy of soft voting

print('Soft Voting:', accuracy_score(y_test, soft_predictions))

      
#  ========================= ********* ===================================


# Objective: 2.	Most cancers form a lump called a tumour. But not all lumps are cancerous. Doctors extract a sample from the lump and examine it to find out if it’s cancer or not. Lumps that are not cancerous are called benign (be-NINE). Lumps that are cancerous are called malignant (muh-LIG-nunt). Obtaining incorrect results (false positives and false negatives) especially in a medical condition such as cancer is dangerous. So, perform Bagging, Boosting, Stacking, and Voting algorithms to increase model performance and provide your insights in the documentation

import pandas as pd
tumor = pd.read_csv('D:\Data Set\Tumor_Ensemble.csv')
tumor.columns
tumor.head()
tumor.info()

# input & output split

predictors = tumor.loc[:,tumor.columns!='diagnosis']
type(predictors)
target = tumor['diagnosis']
type(target)

#Train test partition of the data

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size=0.2, random_state=0)


from sklearn import tree

clftree = tree.DecisionTreeClassifier()

from sklearn.ensemble import BaggingClassifier

bag_clf = BaggingClassifier(base_estimator=clftree, n_estimators=500, bootstrap=True, n_jobs=1, random_state=42)
bag_clf.fit(x_train, y_train)


from sklearn.metrics import accuracy_score, confusion_matrix


# Evaluation on Testing data

confusion_matrix(y_test, bag_clf.predict(x_test))
accuracy_score(y_test, bag_clf.predict(x_test))

#Evaluation on Training data

confusion_matrix(y_train, bag_clf.predict(x_train))
accuracy_score(y_train, bag_clf.predict(x_train))

   # 2 Boosting
   
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(learning_rate=0.02, n_estimators=500)
ada_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

#Evaluation on testing data

confusion_matrix(y_test, ada_clf.predict(x_test))
accuracy_score(y_test, ada_clf.predict(x_test))

#Evaluation on training data
confusion_matrix(y_train, ada_clf.predict(x_train))
accuracy_score(y_train, ada_clf.predict(x_train))
   
# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

gb_clf = GradientBoostingClassifier()
gb_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(y_test, gb_clf.predict(x_test))
accuracy_score(y_test, gb_clf.predict(x_test))
    
#Hyper parameters

glb_clf2 = GradientBoostingClassifier(learning_rate=0.02, n_estimators=10000, max_depth=1)
glb_clf2.fit(x_train, y_train)

# accuracy
from sklearn.metrics import accuracy_score, confusion_matrix

# evaluation on testing data
confusion_matrix(y_test, glb_clf2.predict(x_test))
accuracy_score(y_test, glb_clf2.predict(x_test))

# evaluation on training data

confusion_matrix(y_train, glb_clf2.predict(x_train))
accuracy_score(y_train, glb_clf2.predict(x_train))

# XG Boosting

import xgboost as xgb

xgb_clf = xgb.XGBClassifier(max_depth=5, n_estimators=10000, learning_rate=0.3, n_jobs=-1)

# n_jobs - no.of parallel threads used to run xg boost
# learning_rate(float) - Boosting learning rate (xgb's "eta")

xgb_clf.fit(x_train, y_train)

# evaluation on test data
confusion_matrix(y_test, xgb_clf.predict(x_test))
accuracy_score(y_test, xgb_clf.predict(x_test))

# evaluation on train data
confusion_matrix(y_train, xgb_clf.predict(x_train))
accuracy_score(y_train, xgb_clf.predict(x_train))

xgb.plot_importance(xgb_clf)

xgb_clf = xgb.XGBClassifier(n_estimators=500, learning_rate=0.1, random_state=42)

#n = range(3,10,2)
param_test1 = {'max_depth': range(3,10,2),'gamma':[0.1,0.2,0.3], 'subsample':[0.8,0.9],'colsample_bytree':[0.8,0.9], 'rag_alpha':[1e-2]}

# 3 Grid Search

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(xgb_clf, param_test1, n_jobs=-1, cv=5, scoring='accuracy')

grid_search.fit(x_train, y_train)
cv_xg_clf = grid_search.best_estimator_

#Evaluate on Testing data with model with hyper parameter

accuracy_score(y_test, cv_xg_clf.predict(x_test))

# 5. Stacking

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
# from sklearn SVC import LinearSVC
import warnings
warnings.filterwarnings('ignore')

from mlxtend.classifier import StackingClassifier
# Define base learners

myclf1 = KNeighborsClassifier(n_neighbors=1)
myclf2 = DecisionTreeClassifier(max_depth=4, random_state=123456)
myclf3 = MLPClassifier(hidden_layer_sizes =(100, ), solver='lbfgs', random_state=123456)

# Defining meta model

lr = LogisticRegression()

# Creating stacking classifier with above models


stackingclf = StackingClassifier(classifiers=[myclf1,myclf2,myclf3], use_probas=True, meta_classifier=lr)

print('Doing 3 fold cross validation here:\n')

for iterclf, iterabel in zip([myclf1,myclf2,myclf3,stackingclf], ['K-Nearest Neighbors Model','Decision Tree Classifier','MLP Classifier']):
    
    scores = cross_val_score(iterclf, x_train, y_train, cv=3, scoring='accuracy')
    print('Accuracy: %0.3f(+/-%0.3f) [%s]' %(scores.mean(), scores.std(), iterabel))


#  5. Voting


from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Instatiate the learners (Classifiers)

learner_1 = KNeighborsClassifier(n_neighbors=5)
learner_2 = Perceptron(tol=1e-2, random_state=0)
learner_3 = SVC(gamma=0.001)


# Instantiate the voting classifier

voting = VotingClassifier([('KNN',learner_1),
                           ('Prc',learner_2),
                           ('svm', learner_3)])
 

# Fit Classifier with training data

voting.fit(x_train, y_train)

#Predict the most voted class

hard_predictions = voting.predict(x_test)

# Accuracy of hard voting

print('Hard Voting:', accuracy_score(y_test, hard_predictions))

# 5.1 Soft Voting

# Instatiate the learners (classifier)                         

learner_4 = DecisionTreeClassifier(max_depth=4, random_state=123456)
learner_5 = GaussianNB()
learner_6 = SVC(gamma=0.001, probability=True)


# Instantiating the voting classifier

voting = VotingClassifier([('KNN',learner_4),
                          ('NB',learner_5),
                          ('SVM', learner_6)], voting= 'soft')


# Fit classifier with the training data

voting.fit(x_train, y_train)
learner_4.fit(x_train, y_train)
learner_5.fit(x_train, y_train)
learner_6.fit(x_train, y_train)


# Predict the most probable class

soft_predictions = voting.predict(x_test)


# Get the base learner predictions

prediction_4 = learner_4.predict(x_test)
prediction_5 = learner_5.predict(x_test)
prediction_6 = learner_6.predict(x_test)


# Accuracies of base learners

print('L4:', accuracy_score(y_test, prediction_4))
print('L5:', accuracy_score(y_test, prediction_5))
print('L6:', accuracy_score(y_test, prediction_6))

# Accuracy of soft voting

print('Soft Voting:', accuracy_score(y_test, soft_predictions))


 # ========================== ****************** ================
 
 # Objective: 3.	A sample of global companies and their ratings are given for the cocoa bean production along with the location of the beans being used. Identify the important features in the analysis and accurately classify the companies based on their ratings and draw insights from the data. Build ensemble models such as Bagging, Boosting, Stacking, and Voting on the dataset given. 

import pandas as pd
coca_rating = pd.read_excel('D:\Data Set\Coca_Rating_Ensemble.xlsx')
coca_rating.columns
coca_rating.head()
coca_rating.info()

# input & output split

predictors = tumor['Rating']
type(predictors)
target = tumor['Company']
type(target)

#Train test partition of the data

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size=0.2, random_state=0)

x_train
y_train
from sklearn import tree

clftree = tree.DecisionTreeClassifier()

from sklearn.ensemble import BaggingClassifier

bag_clf = BaggingClassifier(base_estimator=clftree, n_estimators=500, bootstrap=True, n_jobs=1, random_state=42)
bag_clf.fit(x_train, y_train)


from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing data

confusion_matrix(y_test, bag_clf.predict(x_test))
accuracy_score(y_test, bag_clf.predict(x_test))

#Evaluation on Training data

confusion_matrix(y_train, bag_clf.predict(x_train))
accuracy_score(y_train, bag_clf.predict(x_train))

   # 2 Boosting
   
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(learning_rate=0.02, n_estimators=500)
ada_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

#Evaluation on testing data

confusion_matrix(y_test, ada_clf.predict(x_test))
accuracy_score(y_test, ada_clf.predict(x_test))

#Evaluation on training data
confusion_matrix(y_train, ada_clf.predict(x_train))
accuracy_score(y_train, ada_clf.predict(x_train))
   
# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

gb_clf = GradientBoostingClassifier()
gb_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(y_test, gb_clf.predict(x_test))
accuracy_score(y_test, gb_clf.predict(x_test))
    
#Hyper parameters

glb_clf2 = GradientBoostingClassifier(learning_rate=0.02, n_estimators=10000, max_depth=1)
glb_clf2.fit(x_train, y_train)

# accuracy
from sklearn.metrics import accuracy_score, confusion_matrix

# evaluation on testing data
confusion_matrix(y_test, glb_clf2.predict(x_test))
accuracy_score(y_test, glb_clf2.predict(x_test))

# evaluation on training data

confusion_matrix(y_train, glb_clf2.predict(x_train))
accuracy_score(y_train, glb_clf2.predict(x_train))

# XG Boosting

import xgboost as xgb

xgb_clf = xgb.XGBClassifier(max_depth=5, n_estimators=10000, learning_rate=0.3, n_jobs=-1)

# n_jobs - no.of parallel threads used to run xg boost
# learning_rate(float) - Boosting learning rate (xgb's "eta")

xgb_clf.fit(x_train, y_train)

# evaluation on test data
confusion_matrix(y_test, xgb_clf.predict(x_test))
accuracy_score(y_test, xgb_clf.predict(x_test))

# evaluation on train data
confusion_matrix(y_train, xgb_clf.predict(x_train))
accuracy_score(y_train, xgb_clf.predict(x_train))

xgb.plot_importance(xgb_clf)

xgb_clf = xgb.XGBClassifier(n_estimators=500, learning_rate=0.1, random_state=42)

#n = range(3,10,2)
param_test1 = {'max_depth': range(3,10,2),'gamma':[0.1,0.2,0.3], 'subsample':[0.8,0.9],'colsample_bytree':[0.8,0.9], 'rag_alpha':[1e-2]}

# 3 Grid Search

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(xgb_clf, param_test1, n_jobs=-1, cv=5, scoring='accuracy')

grid_search.fit(x_train, y_train)
cv_xg_clf = grid_search.best_estimator_

#Evaluate on Testing data with model with hyper parameter

accuracy_score(y_test, cv_xg_clf.predict(x_test))

# 5. Stacking

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
# from sklearn SVC import LinearSVC
import warnings
warnings.filterwarnings('ignore')

from mlxtend.classifier import StackingClassifier
# Define base learners

myclf1 = KNeighborsClassifier(n_neighbors=1)
myclf2 = DecisionTreeClassifier(max_depth=4, random_state=123456)
myclf3 = MLPClassifier(hidden_layer_sizes =(100, ), solver='lbfgs', random_state=123456)

# Defining meta model

lr = LogisticRegression()

# Creating stacking classifier with above models


stackingclf = StackingClassifier(classifiers=[myclf1,myclf2,myclf3], use_probas=True, meta_classifier=lr)

print('Doing 3 fold cross validation here:\n')

for iterclf, iterabel in zip([myclf1,myclf2,myclf3,stackingclf], ['K-Nearest Neighbors Model','Decision Tree Classifier','MLP Classifier']):
    
    scores = cross_val_score(iterclf, x_train, y_train, cv=3, scoring='accuracy')
    print('Accuracy: %0.3f(+/-%0.3f) [%s]' %(scores.mean(), scores.std(), iterabel))


#  5. Voting


from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Instatiate the learners (Classifiers)

learner_1 = KNeighborsClassifier(n_neighbors=5)
learner_2 = Perceptron(tol=1e-2, random_state=0)
learner_3 = SVC(gamma=0.001)


# Instantiate the voting classifier

voting = VotingClassifier([('KNN',learner_1),
                           ('Prc',learner_2),
                           ('svm', learner_3)])
 

# Fit Classifier with training data

voting.fit(x_train, y_train)

#Predict the most voted class

hard_predictions = voting.predict(x_test)

# Accuracy of hard voting

print('Hard Voting:', accuracy_score(y_test, hard_predictions))

# 5.1 Soft Voting

# Instatiate the learners (classifier)                         

learner_4 = DecisionTreeClassifier(max_depth=4, random_state=123456)
learner_5 = GaussianNB()
learner_6 = SVC(gamma=0.001, probability=True)


# Instantiating the voting classifier

voting = VotingClassifier([('KNN',learner_4),
                          ('NB',learner_5),
                          ('SVM', learner_6)], voting= 'soft')


# Fit classifier with the training data

voting.fit(x_train, y_train)
learner_4.fit(x_train, y_train)
learner_5.fit(x_train, y_train)
learner_6.fit(x_train, y_train)


# Predict the most probable class

soft_predictions = voting.predict(x_test)


# Get the base learner predictions

prediction_4 = learner_4.predict(x_test)
prediction_5 = learner_5.predict(x_test)
prediction_6 = learner_6.predict(x_test)


# Accuracies of base learners

print('L4:', accuracy_score(y_test, prediction_4))
print('L5:', accuracy_score(y_test, prediction_5))
print('L6:', accuracy_score(y_test, prediction_6))

# Accuracy of soft voting

print('Soft Voting:', accuracy_score(y_test, soft_predictions))

# ======================== ************************* ===================

# Objective: To build an ensemble model to classify the user’s password strength

import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score



import pandas as pd
password = pd.read_excel('/content/Ensemble_Password_Strength.xlsx')
password.columns
password.head()

password.characters_strength.unique()

# checking null values
password.isnull().sum()

import seaborn as sns
sns.countplot(password['characters_strength'])

# Convert our entire data into the format of NumPy array
password_tuple = np.array(password)

import random
random.shuffle(password_tuple)

password.characters_strength=password.characters_strength.astype(str)

x = [labels[0] for labels in password_tuple]
y = [labels[1] for labels in password_tuple]

def word_split(inputs):
    character=[]
    for i in inputs:
        character.append(i)
    return character

word_split('kzde5577')

# vectorizing the string (password)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(tokenizer=word_split)
X = vectorizer.fit_transform(x)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.linear_model import LogisticRegression

rfc = RandomForestClassifier(criterion = 'entropy', random_state = 0)
rfc.fit(X_train, y_train)

# prediction on test data
y_pred = rfc.predict(X_test)

# accuracy on test data
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: '+str(round(accuracy*100,2))+'%')

