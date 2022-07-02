# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:48:52 2022

@author: nathl
"""

import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
import shap


application = pd.read_pickle('sample.pkl')
application = application.reset_index(drop=True)
model = load('lgbm_model.joblib')

list_columns = application.columns.to_list()
list_columns.remove('SK_ID_CURR')
list_columns.remove('TARGET')

data = application[list_columns]

# compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(data)


id_client = st.sidebar.selectbox(label='Client ID', options=application['SK_ID_CURR'])
index_client = application[application['SK_ID_CURR']==id_client].index.tolist()

# multioutput_decision_plot the 20 variables explaination for the first individu to obtain 1
st.subheader('Multioutput decision plot')
multioutput_decision_plot = shap.multioutput_decision_plot(explainer.expected_value,
                               shap_values,
                               row_index = index_client[0],
                               feature_names=data.columns.tolist(),
                               highlight = [1]
                               )
shap.initjs()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(multioutput_decision_plot)


# List of the 20 variables that influence the result 
feature_names = data.columns

rf_resultX = pd.DataFrame(shap_values[1], columns = feature_names)

vals = np.abs(rf_resultX.values).mean(0)

shap_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                  columns=['col_name','feature_importance_vals'])
shap_importance.sort_values(by=['feature_importance_vals'],
                               ascending=False, inplace=True)
shap_most_importance = shap_importance.head(20)
col_shap_most_importance = shap_most_importance['col_name'].tolist()

# Clients data
application.loc[application['SK_ID_CURR'] == id_client, col_shap_most_importance]
    


