# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 10:24:12 2022

@author: nathl
"""
from function import prepa_data, clean_data, norm_data
import os.path
from joblib import load, dump
import shap
import numpy as np
import pandas as pd
import pickle

# Folder with file to trait
folder_path = os.path.join("/Users", 'nathl', 'Documents',
                           'FormationDataScientist', 'OpenClassrooms', 
                           'Projet7', '1-Mission', 'PretADepenser')

# Import model
model = load(os.path.join('fastapi-backend','lgbm_model.joblib'))

## threshold calculate for this model
threshold = 0.18

# Pretraitement & consolidation with ategorical columns encode
df = prepa_data(folder_path)


# Non significative columns, N/A & Inf traitement
df = clean_data(df)

# Facultative if not Bigdata
sample = df.sample(frac =.001)

# Normalize numerical columns (not sample data if not bigdata)
sample_norm = norm_data(sample)

# List of columns use in the model
list_columns = sample_norm.columns.to_list()
list_columns.remove('SK_ID_CURR')
list_columns.remove('TARGET')

# compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample_norm[list_columns])

# List of the 20 variables that influence the result 
feature_names = sample_norm[list_columns].columns

probability = model.predict_proba(sample_norm[list_columns])
prediction = np.where(probability[:,1] < threshold, 0, 1)

# Dico with key = index_client and values tuple of most hese important variables
col_shap_most_importance_dic = {}

for i in range (0,len(prediction)):
    
    rf_resultX = pd.DataFrame(shap_values[prediction[i]], columns = feature_names)
    
    vals = np.abs(rf_resultX.filter(
        items=[i],
        axis=0).values).mean(0)
    
    shap_importance = pd.DataFrame(
        list(zip(feature_names, vals)),
        columns=['col_name','feature_importance_vals'])

    shap_importance.sort_values(by=['feature_importance_vals'],
                                   ascending=False, inplace=True)
    shap_most_importance = shap_importance.head(20)
    col_shap_most_importance = shap_most_importance['col_name'].tolist()
    col_shap_most_importance = tuple(col_shap_most_importance)
    col_shap_most_importance_dic[i] = col_shap_most_importance


# Import columns descriptions to create a dict
columns_descritions = pd.read_csv(os.path.join(
    folder_path, 'HomeCredit_columns_description.csv'), sep=',',encoding= 'unicode_escape')

columns_descritions = columns_descritions.drop_duplicates(subset=("Row"), keep='first')

columns_descritions = columns_descritions[["Row","Description"]]

columns_descrition_dic = columns_descritions.set_index("Row").T.to_dict('list')

columns_descrition_dic.update({'DAYS_EMPLOYED_PERC' : ["Percentage of days employed per client's age"],
'INCOME_CREDIT_PERC': ["Percentage of credit amount in relation to a client's income"],
'INCOME_PER_PERSON' : ["Income of a client in relation to the number of shares in his care"],
'ANNUITY_INCOME_PERC' :["Percentage of loan annuity to client's income"],
'PAYMENT_RATE' : ["Annual reimbursement rate"],
'APP_CREDIT_PERC': ["Percentage of the amount requested compared to the receipt"],
'PAYMENT_PERC': ["Percentage of credit repayment"],
'PAYMENT_DIFF': ["Percentage of what remains to be paid"],
'DPD' : ["Number of days of late payment"],
'DBD': ["Number of days of advance payment"]})



# save data
sample.to_pickle(os.path.join('streamlit-frontend', 'sample.pkl'))

sample_norm.to_pickle(os.path.join('streamlit-frontend', 'sample_norm.pkl'))

dump(explainer,os.path.join('streamlit-frontend', 'explainer.bz2'), compress=('bz2', 9))

dump(shap_values,os.path.join('streamlit-frontend', 'shap_values.joblib'))

pickle.dump(col_shap_most_importance_dic, open(os.path.join('streamlit-frontend',"col_shap_most_importance_dic.p"), 'wb'))

pickle.dump(columns_descrition_dic, open(os.path.join('streamlit-frontend',"columns_descrition_dic.p"), 'wb'))

sample_norm.to_pickle(os.path.join('fastapi-backend', 'sample_norm.pkl'))