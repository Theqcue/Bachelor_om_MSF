# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd
import seaborn as sn 
from matplotlib import pyplot as plt 
from sklearn.model_selection import train_test_split 
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import Dropout 
from keras.constraints import maxnorm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

Churn_DF = pd.read_pickle("Churn_DF.pkl")  
Prediction_set = pd.read_pickle("Prediction_set.pkl")

dummy_df = pd.get_dummies(Churn_DF)

y = dummy_df.Churn.values
X = dummy_df.drop('Churn', axis = 1)
cols = X.columns

from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
X = pd.DataFrame(mm.fit_transform(X))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.25,random_state=0)


#-------------------- Random forest ----------------------------#

rf = RandomForestClassifier(n_estimators=100, max_depth=20,
                              random_state=42)

rf.fit(X_train, y_train) 
score = rf.score(X_train, y_train)
score2 = rf.score(X_val, y_val)

# ------------------ Evaluering ---------------------------#
y_pred_rf = rf.predict(X_val)
y_pred_rf_train = rf.predict(X_train)


#val

Accuracy_rf = accuracy_score(y_val, y_pred_rf)
Precision_rf = precision_score(y_val, y_pred_rf)
Recall_rf = recall_score(y_val, y_pred_rf)
F1_Score_rf = f1_score(y_val, y_pred_rf)

Eval_Metrics = [Accuracy_rf, Precision_rf, Recall_rf, F1_Score_rf]
Metric_Names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']


#Train

Accuracy_rf_train = accuracy_score(y_train, y_pred_rf_train)
Precision_rf_train = precision_score(y_train, y_pred_rf_train)
Recall_rf_train = recall_score(y_train, y_pred_rf_train)
F1_Score_rf_train = f1_score(y_train, y_pred_rf_train)

Eval_Metrics = [Accuracy_rf_train, Precision_rf_train, Recall_rf_train, F1_Score_rf_train]
Metric_Names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']


#-------------------------Features-------------------#

fi = pd.DataFrame({'feature': list(cols[X_train.columns]),
                   'importance': rf.feature_importances_}).\
                    sort_values('importance', ascending = False)
fi.head()



# ------------------ Neaural network ---------------------------#
#https://medium.com/swlh/building-an-artificial-neural-network-in-less-than-10-minutes-cbe59dbb903c

model = Sequential()
model.add(Dense(68, input_dim=2547, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(rate=0.2))
model.add(Dense(8, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(rate=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss = "binary_crossentropy", optimizer = 'adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=8)

y_pred = model.predict(X_val)
y_pred_train = model.predict(X_train)
y_pred = (y_pred > 0.5)
y_pred_train = (y_pred_train > 0.5)

#val 
y_pred = y_pred.flatten(order='C')
y_pred_train = y_pred_train.flatten(order='C')


Accuracy_val_nn = accuracy_score(y_val, y_pred)
Precision_val_nn = precision_score(y_val, y_pred)
Recall_val_nn = recall_score(y_val, y_pred)
F1_Score_val_nn = f1_score(y_val, y_pred)

Eval_Metrics = [Accuracy_val_nn, Precision_val_nn, Recall_val_nn, F1_Score_val_nn]
Metric_Names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

#Train

Accuracy_train_nn = accuracy_score(y_train, y_pred_train)
Precision_train_nn = precision_score(y_train, y_pred_train)
Recall_train_nn = recall_score(y_train, y_pred_train)
F1_Score_train_nn = f1_score(y_train, y_pred_train)

Eval_Metrics = [Accuracy_train_nn, Precision_train_nn, Recall_train_nn, F1_Score_train_nn]
Metric_Names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']



#----------------------------------- XGBoost --------------------------------------#

xgb_model = xgb.XGBClassifier(max_depth=5, learning_rate=0.08, objective= 'binary:logistic',n_jobs=-1).fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_val)
y_pred_xgb_train = xgb_model.predict(X_train)


#val set

Accuracy_train_xg = accuracy_score(y_val, y_pred_xgb)
Precision_train_xg = precision_score(y_val, y_pred_xgb)
Recall_train_xg = recall_score(y_val, y_pred_xgb)
F1_Score_train_xg = f1_score(y_val, y_pred_xgb)

Eval_Metrics = [Accuracy_train_xg, Precision_train_xg, Recall_train_xg, F1_Score_train_xg]
Metric_Names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

#Training set

Accuracy_train_xg_train = accuracy_score(y_train, y_pred_xgb_train)
Precision_train_xg_train = precision_score(y_train, y_pred_xgb_train)
Recall_train_xg_train = recall_score(y_train, y_pred_xgb_train)
F1_Score_train_xg_train = f1_score(y_train, y_pred_xgb_train)

Eval_Metrics = [Accuracy_train_xg_train, Precision_train_xg_train, Recall_train_xg_train, F1_Score_train_xg_train]
Metric_Names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

#------------------------------Feature importance-------------------#

feature_important = xgb_model.get_booster().get_score(importance_type='gain')
keys = list(feature_important.keys())
values = list(feature_important.values())

df_feature_important = pd.DataFrame.from_dict(feature_important, orient='index')
df_feature_important.reset_index(inplace=True)
df_feature_important.columns = ['index', 'fi']

fi_xgboost = pd.DataFrame({'feature': list(cols[df_feature_important.index]),
                   'importance': df_feature_important.fi}).\
                    sort_values('importance', ascending = False)
fi_xgboost.head()






