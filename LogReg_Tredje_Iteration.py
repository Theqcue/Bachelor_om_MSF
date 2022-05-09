# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


Churn_DF = pd.read_pickle("Churn_DF.pkl")  
Prediction_set = pd.read_pickle("Prediction_set.pkl")

dummy_df = pd.get_dummies(Churn_DF).sort_values(by='Churn', ascending=False) 

y = dummy_df.Churn.values
X = dummy_df.drop('Churn', axis = 1)

cols = X.columns

mm = MinMaxScaler()
X = pd.DataFrame(mm.fit_transform(X))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.25,random_state=0)


#-------------------------------------- Grid search --------------------------#
#https://towardsdatascience.com/grid-search-for-model-tuning-3319b259367e

no_of_churn = np.count_nonzero(y_train == 1)
no_of_notchurn = np.count_nonzero(y_train == 0)

Perc_of_churn=(no_of_churn/no_of_notchurn)*100
Perc_of_no_churn = 100 - Perc_of_churn
w = {0:1, 1:Perc_of_no_churn}

#Grid search other parameters 
clf = LogisticRegression(solver='newton-cg',C=1, penalty='l2')
grid_values = [
  {'penalty': ['l1', 'l2'], 'solver': [ 'liblinear'],'C':[1, 5, 10]},
  {'penalty': ['l2'], 'solver': ['newton-cg','lbfgs'],'C':[1, 5, 10]},
 ]

#Grid search weight
w = [{0:1, 1:Perc_of_no_churn},{0:0.01,1:1.0}, {0:0.01,1:10}, {0:0.01,1:100}, 
     {0:0.001,1:1.0}, {0:0.005,1:1.0}]

grid_values = {"class_weight": w }


grid_clf_acc = GridSearchCV(clf, param_grid = grid_values,scoring = 'recall', verbose=1, cv=3, n_jobs=-1)
grid_clf_acc.fit(X_train, y_train)

print("Best: {0}, using {1}".format(grid_clf_acc.cv_results_['mean_test_score'], grid_clf_acc.best_params_))

#---------------------------------------------- Model ------------------------------------#
w = {0: 0.2, 1: 1.5}
logistic_regression = LogisticRegression(solver='newton-cg',C=0.1, penalty='l2', class_weight=w)
logistic_regression.fit(X_train,y_train)


y_pred=logistic_regression.predict(X_val)
y_train_pred=logistic_regression.predict(X_train)
y_pred_test=logistic_regression.predict(X_test)


precision_val = precision_score(y_val, y_pred)
recall_val = recall_score(y_val, y_pred)
accuracy_val = accuracy_score(y_val, y_pred)
f1_val = f1_score(y_val, y_pred)


precision_train = precision_score(y_train, y_train_pred)
recall_train = recall_score(y_train, y_train_pred)
accuracy_train = accuracy_score(y_train, y_train_pred)
f1_train = f1_score(y_train, y_train_pred)

precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)

Eval_Metrics_test = [accuracy_test, precision_test, recall_test, f1_test]
Eval_Metrics_train = [accuracy_train, precision_train, recall_train, f1_train]
Eval_Metrics = [accuracy_val, precision_val, recall_val, f1_val]
Metric_Names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

scores_val = pd.DataFrame(Metric_Names)
scores_val["train"] = Eval_Metrics_train
scores_val["Val"] = Eval_Metrics
scores_val["Test"] = Eval_Metrics_test

#--------------------------------------- Features ------------------------------------#

weight = logistic_regression.coef_  

coefs=logistic_regression.coef_[0]
top_five = np.argpartition(coefs, -5)[-5:]
low_five = np.argsort(coefs)[0:5]
index = [*top_five, *low_five]

print(cols[index])
print(coefs[index])

data = {'Name':cols[index],
        'Weight-log odds':coefs[index],
        'Weight-odds':np.exp(coefs[index]),
        }
Most_important_featuers = pd.DataFrame(data)

