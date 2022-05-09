# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

dtypesDDA = {'ACCOUNT_TYPE__C': 'string',
                     'AGREEMENT_STATUS__C': 'string',
                     'CAMPAIGNTYPE__C': 'string',
                     'CAMPAIGN_MEDIA__C': 'string',
                     'DIRECT_DEBIT_AGREEMENT_STATUS_REASON__C': 'string',
                     'DIRECT_DEBIT_AGREEMENT_STATUS__C': 'string',
                     'DONOR__C': 'string',
                     'FREQUENCYMC__C': 'string',
                     'FREQUENCY__C': 'string',
                     'SOURCE_CAMPAIGN_CODE__C': 'string',
                     'SOURCE_CAMPAIGN__C': 'string',
                     'SOURCE_OF_SIGN_UP__C': 'string',
                     }

date_cols_DDA = ['CREATEDDATE', 
                 'END_DATE__C',
                 "START_DATE__C"]


dtypeContact = {'CONTACT_ID_18__C': 'string',
                     'ACTIVE_DDA__C': 'string',
                     'COUNTRYCODE__C': 'string',
                     'DDASTATUS__C': 'string',
                     'DONORCURRENTSTATUS__C': 'string',
                     'DONOR_STATUS__C': 'string',
                     'FIRSTCAMPAIGN__C': 'string',
                     'FIRSTPAYMENTCAMPAIGN__C': 'string',
                     'GENDER__C': 'string',
                     'HARD_BOUNCE_REASON__C': 'string',
                     'ID': 'string',
                     'MAILINGCITY': 'string',
                     'MAILINGCOUNTRY': 'string',
                     'MAILINGCOUNTRYCODE': 'string',
                     'MAILINGPOSTALCODE': 'string',
                     'MAILINGSTATE': 'string',
                     'MAILINGSTATECODE': 'string',
                     'MOBPAYSUB_STATUS__C': 'string',
                     'RECORDTYPE__C': 'string',
                     'SEND_JOURNAL_AS__C': 'string',
                     'STRIPE_CUSTOMER_ID_CONTACT__C': 'string',
                     'STRIPE_STATUS_CONTACT__C': 'string',
                     'TYPE__C': 'string'}

date_cols_contact = ['BIRTHDATE',
                 'GDPRDELETED__C',
                 'FIRSTDONATIONDATE__C',
                 'LAST_TM_DATE__C',
                 ]



col_list_Contact = ["TOTAL_AMOUNT_DONATED1__C", "TOTAL_AMOUNT_DONATED_REGULAR__C", "GDPRDELETED__C", "AGE__C", "GENDER__C", "ID", "BIRTHDATE",'ALL_COMMUNICATION_OPT_OUT__C', "PHYSICAL_MAIL_OPT_OUT__C", "NO_OF_DONATION__C", "NO_SMS_DONATIONER__C", "NO_ONLINE_DONATIONER__C", "MAILINGCITY", "MAILINGPOSTALCODE", "HASOPTEDOUTOFEMAIL", 'FIRSTCAMPAIGN__C', 'FIRSTDONATIONDATE__C', 'FIRSTPAYMENTCAMPAIGN__C', 'LAST_TM_DATE__C', 'MAJOR_DONOR__C', 'TAGER_IKKE_TELEFONEN__C', 'TELEMARKETING_OPT_OUT__C', 'TAX_DEDUCTION__C']
col_list_DDA = [ "SIGNED_UP_FOR_DIRECT_DEBIT_ON__C","DONOR__C","CREATEDDATE","ID", "FREQUENCY__C", "AMOUNT__C","START_DATE__C", "DIRECT_DEBIT_AGREEMENT_STATUS__C", "END_DATE__C"]

AllDDA = pd.read_csv('AllDDA.csv', encoding= 'unicode_escape', usecols=col_list_DDA, parse_dates=date_cols_DDA)
AllContact = pd.read_csv('AllContacts.csv', encoding= 'unicode_escape', usecols=col_list_Contact, parse_dates=date_cols_contact) 

AllDDA['CREATEDDATE'] =  AllDDA['CREATEDDATE'].dt.tz_localize(None)

sn.histplot(data=AllDDA, x="DIRECT_DEBIT_AGREEMENT_STATUS__C")
plt.show()


print(type(AllDDA['START_DATE__C'][0].tzinfo))
print(type(AllDDA['CREATEDDATE'][0].tzinfo))

AllContact['One_off_donation_total'] = (AllContact['TOTAL_AMOUNT_DONATED1__C'] - AllContact['TOTAL_AMOUNT_DONATED_REGULAR__C'])

#Defining churn
today = datetime.today()
one_year_behind_now = today - relativedelta(years=1)

AllDDA['days_start_end'] = (AllDDA['END_DATE__C'] - AllDDA['START_DATE__C']).dt.days
AllDDA['days_Created_end'] = (AllDDA['END_DATE__C'] - AllDDA['CREATEDDATE']).dt.days
AllDDA['Prediction_set'] = (AllDDA['CREATEDDATE'] > one_year_behind_now) & (AllDDA['days_Created_end'].isnull())

All_Terminated_dda = AllDDA[AllDDA['days_Created_end'].notnull()]

bins= [-200,0,200,400,600,800,1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600]
labels = ['-200:-1','0:199','200:399','400:599','600:799','800:999','1000:1199','1200:1399','1400:1599','1600:1799', '1800:1999', '2000:2199', '2200: 2399', '2400:2599' ]

All_Terminated_dda['Time_group'] = pd.cut(All_Terminated_dda['days_Created_end'], bins=bins, labels=labels, right=False)

sn.histplot(data=All_Terminated_dda, y="Time_group").set(title='All terminated DDA grouped in days from start to end')
plt.show()

# create a list of our conditions
conditions = [
    (AllDDA['days_Created_end'] <= 365),
    (AllDDA['Prediction_set']),
    ((AllDDA['days_Created_end'] > 365) | (AllDDA['days_Created_end'].isnull()))
    ]

# create a list of the values we want to assign for each condition
values = ['Churn', 'Prediction_set','No_churn']

# create a new column and use np.select to assign values to it using our lists as arguments
AllDDA['Churn'] = np.select(conditions, values)

sn.histplot(data=AllDDA, x="Churn")
plt.show()

conditions = [
    (AllDDA['FREQUENCY__C'] == 'Annually'),
    (AllDDA['FREQUENCY__C'] == 'Quarterly' ),
    (AllDDA['FREQUENCY__C'] == 'Monthly'),
    (AllDDA['FREQUENCY__C'] == 'Bi-annually')
    ]

# create a list of the values we want to assign for each condition
values = [AllDDA['AMOUNT__C'] / 12, AllDDA['AMOUNT__C'] / 4, AllDDA['AMOUNT__C'],AllDDA['AMOUNT__C']/6 ]
AllDDA['Monthly_amount'] = np.select(conditions, values)
bins= [0,50,100,150,200,250, 300, 400, 500, 1000, 2000, 10000, 20000]
labels = ['0:49','50:99','100:149','150:199','200:249','250:299','300:399','400:499','500:999','1000:1999','2000:10000','10000:20000']
AllDDA['Amount_grouped'] = pd.cut(AllDDA['Monthly_amount'], bins=bins, labels=labels, right=False)

sn.histplot(data=AllDDA, y="Amount_grouped")
plt.show()
sn.histplot(binwidth=0.5, y="Amount_grouped", hue="Churn", data=AllDDA, stat="count", multiple="stack")


#AllDDA['LAST_TM_DATE__C'] = pd.to_numeric(pd.to_datetime(Churn_DF['LAST_TM_DATE__C']))
AllContact['Days_since_LAST_TM'] = (today - AllContact['LAST_TM_DATE__C']).dt.days                 
AllContact['has_Been_Called_TM'] = (AllContact['Days_since_LAST_TM']).notnull()

# ----------------------------------- Merging + cleaning --------------------------------#

Both_DFs = pd.merge(AllDDA,AllContact,left_on=['DONOR__C'],right_on=['ID'], how='inner')
removedfirstCampaign = Both_DFs.loc[Both_DFs['FIRSTCAMPAIGN__C'] == '7010J00000130HJQAY']

removedfirstPaymentCampaign = Both_DFs.loc[Both_DFs['FIRSTPAYMENTCAMPAIGN__C'] == '7016M000001yYSgQAM']
#Fjerner alle postnumre som i hvertfald ikke er fra DK. + ugyldige postnume.
More4postcode = Both_DFs.loc[Both_DFs['MAILINGPOSTALCODE'].map(str).apply(len) != 4]
minusOneOffDonations = Both_DFs.loc[Both_DFs['One_off_donation_total'] < 0] 


Both_DFs = Both_DFs.drop(removedfirstCampaign.index)
Both_DFs = Both_DFs.drop(removedfirstPaymentCampaign.index)
Both_DFs = Both_DFs.drop(More4postcode.index)
Both_DFs = Both_DFs.drop(minusOneOffDonations.index)

Churn_only = Both_DFs.loc[Both_DFs['Churn'] == 'Churn']
No_Churn_only = Both_DFs.loc[Both_DFs['Churn'] == 'No_churn']
Prediction = Both_DFs.loc[Both_DFs['Churn'] == 'Prediction_set']

Churn_Monthly_amount = Churn_only['Monthly_amount'].mean()
No_churn_Monthly_amount = No_Churn_only['Monthly_amount'].mean()
Prediction_Monthly_amount = Prediction['Monthly_amount'].mean()

All_Terminated_dda_churn= Churn_only[Churn_only['days_Created_end'].notnull()]
All_Terminated_dda_No_Churn= No_Churn_only[No_Churn_only['days_Created_end'].notnull()]

All_Terminated_dda_churn= Churn_only[Churn_only['days_Created_end'].notnull()]
All_Terminated_dda_No_Churn= No_Churn_only[No_Churn_only['days_Created_end'].notnull()]

Average_length_of_deal_churn = All_Terminated_dda_churn['days_Created_end'].mean()
Average_length_of_deal_no_churn =All_Terminated_dda_No_Churn['days_Created_end'].mean()


Chosen_DF = Both_DFs.drop([
                           "TOTAL_AMOUNT_DONATED1__C",
                           "MAILINGCITY",
                           "TOTAL_AMOUNT_DONATED_REGULAR__C",
                           "Amount_grouped", 
                           "LAST_TM_DATE__C",
                           "AMOUNT__C",
                           "FIRSTDONATIONDATE__C" ,
                           "BIRTHDATE",
                           "days_start_end",
                           "days_Created_end" ,
                           "CREATEDDATE", 
                           "AMOUNT__C", 
                           "DONOR__C", 
                           "ID_x",
                           "SIGNED_UP_FOR_DIRECT_DEBIT_ON__C",
                           "ID_y", 
                           "START_DATE__C", 
                           "DIRECT_DEBIT_AGREEMENT_STATUS__C", 
                           "END_DATE__C"], axis=1)

Churn_DF = Chosen_DF.drop(Chosen_DF[Chosen_DF.Prediction_set == True].index)
Churn_DF = Churn_DF.drop("Prediction_set", axis=1)

Prediction_set = Chosen_DF.query('Prediction_set == True')

Churn_DF.Churn.replace({"Churn":1, "No_churn":0}, inplace = True)

Churn_DF_is_null = Churn_DF.isnull().sum()

dtype = Churn_DF.dtypes

Churn_DF['AGE__C'] = Churn_DF['AGE__C'].fillna(Churn_DF['AGE__C'].mean())
Churn_DF['GENDER__C'] = Churn_DF['GENDER__C'].fillna('unknown')
Churn_DF['FIRSTCAMPAIGN__C'] = Churn_DF['FIRSTCAMPAIGN__C'].fillna('unknown')
Churn_DF['FIRSTPAYMENTCAMPAIGN__C'] = Churn_DF['FIRSTPAYMENTCAMPAIGN__C'].fillna('unknown')
Churn_DF['Days_since_LAST_TM'] = Churn_DF['Days_since_LAST_TM'].fillna(-1)
Churn_DF_is_null = Churn_DF.isnull().sum()

# Removing GDPR records
AllGDPR = Churn_DF.dropna(subset=['GDPRDELETED__C'])
Churn_DF = Churn_DF.drop(AllGDPR.index)
Churn_DF = Churn_DF.drop("GDPRDELETED__C", axis=1)

#Create PKL
Churn_DF.to_pickle("Churn_DF.pkl")
Prediction_set.to_pickle("Prediction_set.pkl")
