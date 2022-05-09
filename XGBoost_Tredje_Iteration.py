# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import time
import tracemalloc
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

#--------------------------------- Grid search ----------------------------------------#

parameters = {
    'max_depth':[5, 10],
    'n_estimators': [100, 1000],
    'learning_rate': [0.1, 0.01]
}
estimator = XGBClassifier(
    objective= 'binary:logistic',
    n_jobs=-1,
    eval_metric = 'logloss', 
    use_label_encoder=False
)

grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring = 'recall',
    n_jobs = -1,
    cv = 3,
    verbose=10
)


grid_search.fit(X_train, y_train)

#----------------------------------------------- Model ------------------------------#

tracemalloc.start()
time_start = time.perf_counter()
xgb_model = XGBClassifier(scale_pos_weight=12,
                          	max_depth=4,
                          	learning_rate=0.01,
                          	objective= 'binary:logistic',
                          	n_jobs=-1, eval_metric = 'logloss',
                          	use_label_encoder=False, verbose=10).fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_val)
y_pred_xgb_train = xgb_model.predict(X_train)

time_elapsed = (time.perf_counter() - time_start)
memMb=tracemalloc.get_traced_memory()
print ('time:' + str(time_elapsed))
print ('memory in bytes, current, peak:' + str(memMb))
tracemalloc.stop()


Accuracy_train_xg = accuracy_score(y_val, y_pred_xgb)
Precision_train_xg = precision_score(y_val, y_pred_xgb)
Recall_train_xg = recall_score(y_val, y_pred_xgb)
F1_Score_train_xg = f1_score(y_val, y_pred_xgb)

Eval_Metrics = [Accuracy_train_xg, Precision_train_xg, Recall_train_xg, F1_Score_train_xg]
Metric_Names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

Accuracy_train_xg_train = accuracy_score(y_train, y_pred_xgb_train)
Precision_train_xg_train = precision_score(y_train, y_pred_xgb_train)
Recall_train_xg_train = recall_score(y_train, y_pred_xgb_train)
F1_Score_train_xg_train = f1_score(y_train, y_pred_xgb_train)
Eval_Metrics_train = [Accuracy_train_xg_train, Precision_train_xg_train, Recall_train_xg_train, F1_Score_train_xg_train]

y_pred_test=xgb_model.predict(X_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)

Eval_Metrics_test = [accuracy_test, precision_test, recall_test, f1_test]
scores_val = pd.DataFrame(Metric_Names)
scores_val["train"] = Eval_Metrics_train
scores_val["val"] = Eval_Metrics
scores_val["Test"] = Eval_Metrics_test


#------------------------------- Feature ------------------------------------#
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

import lime
from lime import lime_tabular

explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=cols,
    discretize_continuous=False,
    mode='classification'
)


exp = explainer.explain_instance(
    data_row=X_val.iloc[0], # 14 - 2544
    predict_fn=xgb_model.predict_proba, 
    num_features=30000
)
total_List = pd.DataFrame(exp.as_list())
total_List.columns = ['feature', 'weight1']

for i in range(0, 3000, 99):    
    exp = explainer.explain_instance(
        data_row=X_val.iloc[i], # 14 - 2544
        predict_fn=xgb_model.predict_proba, 
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
