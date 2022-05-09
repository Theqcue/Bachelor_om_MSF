# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
import time
import tracemalloc
from lime import lime_tabular

Churn_DF = pd.read_pickle("Churn_DF.pkl")  
Prediction_set = pd.read_pickle("Prediction_set.pkl")

dummy_df = pd.get_dummies(Churn_DF)
y = dummy_df.Churn.values
X = dummy_df.drop('Churn', axis = 1)
cols = X.columns

mm = MinMaxScaler()
X = pd.DataFrame(mm.fit_transform(X))
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.25)


#------------------------- random grid search -------------------#
#https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(random_grid)

#https://www.geeksforgeeks.org/monitoring-memory-usage-of-a-running-python-program/
BRFC = BalancedRandomForestClassifier(random_state=2,n_jobs=-1)
rf_random = RandomizedSearchCV(estimator = BRFC, param_distributions = random_grid, n_iter = 5 , cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train, y_train)
rf_random.best_params_


#------------------------- Model -------------------#

tracemalloc.start()
time_start = time.perf_counter()
BRFC_best = BalancedRandomForestClassifier(n_estimators =1000,min_samples_split= 10, min_samples_leaf=15, max_features= 500, max_depth= 15, bootstrap= True,n_jobs=-1, verbose=10)
BRFC_best.fit(X_train, y_train)
y_pred_rf = BRFC_best.predict(X_val)

y_pred_rf_train = BRFC_best.predict(X_train)

time_elapsed = (time.perf_counter() - time_start)
memMb=tracemalloc.get_traced_memory()
print ('time:' + str(time_elapsed))
print ('memory in bytes, current, peak:' + str(memMb))
tracemalloc.stop()



Accuracy_rf = accuracy_score(y_val, y_pred_rf)
Precision_rf = precision_score(y_val, y_pred_rf)
Recall_rf = recall_score(y_val, y_pred_rf)
F1_Score_rf = f1_score(y_val, y_pred_rf)

Eval_Metrics = [Accuracy_rf, Precision_rf, Recall_rf, F1_Score_rf]
Metric_Names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']


Accuracy_rf_train = accuracy_score(y_train, y_pred_rf_train)
Precision_rf_train = precision_score(y_train, y_pred_rf_train)
Recall_rf_train = recall_score(y_train, y_pred_rf_train)
F1_Score_rf_train = f1_score(y_train, y_pred_rf_train)

Eval_Metrics_train = [Accuracy_rf_train, Precision_rf_train, Recall_rf_train, F1_Score_rf_train]


y_pred_test=BRFC_best.predict(X_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)

Eval_Metrics_test = [accuracy_test, precision_test, recall_test, f1_test]

scores_val = pd.DataFrame(Metric_Names)
scores_val["train"] = Eval_Metrics_train
scores_val["val"] = Eval_Metrics
scores_val["Test"] = Eval_Metrics_test



#------------------------- Features -------------------#
#https://towardsdatascience.com/introduction-to-random-forest-classifiers-9a3b8d8d3fa7
fi = pd.DataFrame({'feature': list(cols[X_train.columns]),
                   'importance': BRFC_best.feature_importances_}).\
                    sort_values('importance', ascending = False)
fi.head()


#https://towardsdatascience.com/lime-how-to-interpret-machine-learning-models-with-python-94b0e7e4432e
#https://github.com/marcotcr/lime
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=cols,
    discretize_continuous=False,
    mode='classification'
)

exp = explainer.explain_instance(
    data_row=X_val.iloc[0], # 14 - 2544
    predict_fn=BRFC_best.predict_proba, 
    num_features=30000
)

total_List = pd.DataFrame(exp.as_list())
total_List.columns = ['feature', 'weight1']

for i in range(0, 3000, 100):    
    exp = explainer.explain_instance(
        data_row=X_val.iloc[i], # 14 - 2544
        predict_fn=BRFC_best.predict_proba, 
        num_features=30000
    )
    exp = pd.DataFrame(exp.as_list())
    stri = 'weight' + str(i)
    strif = 'feature' + str(i)
    print(stri)
    exp.columns = [strif, stri]
    exp = exp.drop(strif, axis=1)
    total_List = pd.concat([total_List, exp.reindex(total_List.index)], axis=1)
total_List['mean'] = total_List.mean(axis=1)
total_List = total_List.reset_index()
total_List = total_List[total_List['feature'].str.contains("Days_since_LAST_TM") | 
                        total_List['feature'].str.contains("NO_OF_DONATION__C") |
                        total_List['feature'].str.contains("AGE__C") |
                        total_List['feature'].str.contains("70120000000cAXbAAM") |
                        total_List['feature'].str.contains("has_Been_Called_TM") |
                        total_List['feature'].str.contains("Monthly_amount") |
                        total_List['feature'].str.contains("NO_ONLINE_DONATIONER__C") |
                        total_List['feature'].str.contains("TELEMARKETING_OPT_OUT__C") |
                        total_List['feature'].str.contains("PHYSICAL_MAIL_OPT_OUT__C") |
                        total_List['feature'].str.contains("HASOPTEDOUTOFEMAIL") |
                        total_List['feature'].str.contains("TAGER_IKKE_TELEFONEN__C") |
                        total_List['feature'].str.contains("TAX_DEDUCTION__C") |
                        total_List['feature'].str.contains("NO_SMS_DONATIONER__C") |
                        total_List['feature'].str.contains("ALL_COMMUNICATION_OPT_OUT__C") |
                        total_List['feature'].str.contains("FREQUENCY__C_Monhtly") |
                        total_List['feature'].str.contains("FREQUENCY__C_Quarterly") |
                        total_List['feature'].str.contains("One_off_donation_total") |
                        total_List['feature'].str.contains("MAJOR_DONOR__C") 
                        ]

total_List = total_List[['feature','mean']]