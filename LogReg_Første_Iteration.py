# -*- coding: utf-8 -*-
"""
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


Churn_DF = pd.read_pickle("Churn_DF.pkl")  
Prediction_set = pd.read_pickle("Prediction_set.pkl")

dummy_df = pd.get_dummies(Churn_DF)

y = dummy_df.Churn.values
X = dummy_df.drop('Churn', axis = 1)
cols = X.columns


mm = MinMaxScaler()
X = pd.DataFrame(mm.fit_transform(X))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.25,random_state=0)



logistic_regression= LogisticRegression()
logistic_regression.fit(X_train,y_train)

y_pred=logistic_regression.predict(X_val)
y_train_pred=logistic_regression.predict(X_train)

weight = logistic_regression.coef_  

#------------------ Features -----------------"

coefs=logistic_regression.coef_[0]
top_five = np.argpartition(coefs, -5)[-5:]
low_five = np.argsort(coefs)[0:5]
index = [*top_five, *low_five]
print(cols[index])
print(coefs[index])

data = {'Name':cols[index],
        'Weight':coefs[index]
        }
Most_important_featuers = pd.DataFrame(data)

#-------------Evalutering -----------#
#val

precision_val = precision_score(y_val, y_pred)
recall_val = recall_score(y_val, y_pred)
accuracy_val = accuracy_score(y_val, y_pred)
f1_val = f1_score(y_val, y_pred)

Eval_Metrics = [accuracy_val, precision_val, recall_val, f1_val]
Metric_Names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

#Train
precision_train = precision_score(y_train, y_train_pred)
recall_train = recall_score(y_train, y_train_pred)
accuracy_train = accuracy_score(y_train, y_train_pred)
f1_train = f1_score(y_train, y_train_pred)

Eval_Metrics = [accuracy_train, precision_train, recall_train, f1_train]
Metric_Names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
