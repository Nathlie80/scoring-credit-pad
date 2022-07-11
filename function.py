# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 08:48:16 2022

@author: nathl
"""

# Library import
import os.path
import re
import gc
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import normalize

# Functions

# 1.   Categorical columns encode
# 2.   Pretraitement & consolidation
# 3.   Non significative columns, N/A & Inf traitement
# 4.   Normalize numerical columns


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns
gc.collect()


# Pretraitement & consolidation
def prepa_data(folder_path):
 # Import & traitement data application_train
    df = pd.read_csv(os.path.join(
        folder_path, 'application_train.csv'), sep=',')

    # Exclusion of CODE_GENDER 'XNA'  from the dataset
    df = df[df['CODE_GENDER'] != 'XNA']

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(
            df[bin_feature],sort = bool)
    gc.collect()

    # Categorical features with One-Hot encode
    df, categorical_columns = one_hot_encoder(df, True)
    gc.collect()

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)

    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    gc.collect()

 # Import & traitement  data bureau & bureau_bal
    bureau = pd.read_csv(os.path.join(
        folder_path, 'bureau.csv'),sep=',')
    bureau_bal = pd.read_csv(os.path.join(
        folder_path, 'bureau_balance.csv'), sep=',')
    
    # Categorical features with One-Hot encode
    bureau_bal, bureau_bal_cat = one_hot_encoder(bureau_bal, True)
    bureau, bureau_cat = one_hot_encoder(bureau, True)
    gc.collect()

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bureau_bal_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bureau_bal_cat:
        bureau_bal_aggregations[col] = ['mean']
    bureau_bal_agg = bureau_bal.groupby('SK_ID_BUREAU').agg(bureau_bal_aggregations)
    bureau_bal_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bureau_bal_agg.columns.tolist()])
    bureau = bureau.join(bureau_bal_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    gc.collect()

    # Bureau and bureau_balance numeric features
    num_aggregations = {
            'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
            'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
            'DAYS_CREDIT_UPDATE': ['mean'],
            'CREDIT_DAY_OVERDUE': ['max', 'mean'],
            'AMT_CREDIT_MAX_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
            'AMT_ANNUITY': ['max', 'mean'],
            'CNT_CREDIT_PROLONG': ['sum'],
            'MONTHS_BALANCE_MIN': ['min'],
            'MONTHS_BALANCE_MAX': ['max'],
            'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }

    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bureau_bal_cat: cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg(
        {**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(
        ['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    
    gc.collect()

    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    
    gc.collect()

    # Join df & bureau_agg
    df = df.join(bureau_agg, how='left', on='SK_ID_CURR')

    # Removal of non-useful variables
    del bureau, bureau_bal, active_agg, active,closed, closed_agg, bureau_agg

 # Import & traitement  data previous_application
    previous_appli = pd.read_csv(os.path.join(
    folder_path, 'previous_application.csv'), sep=',')

    # Categorical features with One-Hot encode
    previous_appli, previous_appli_cat = one_hot_encoder(previous_appli, True)
    gc.collect()

    # Days 365.243 values -> nan
    previous_appli['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    previous_appli['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    previous_appli['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    previous_appli['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    previous_appli['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    gc.collect()

    # Add feature: value ask / value received percentage
    previous_appli['APP_CREDIT_PERC'] = previous_appli['AMT_APPLICATION'] / previous_appli['AMT_CREDIT']

    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }

    # Previous applications categorical features
    cat_aggregations = {}
    for cat in previous_appli_cat:
        cat_aggregations[cat] = ['mean']

    prev_agg = previous_appli.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    gc.collect()

    # Previous Applications: Approved Applications - only numerical features
    approved = previous_appli[previous_appli['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    gc.collect()

    # Previous Applications: Refused Applications - only numerical features
    refused = previous_appli[previous_appli['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    gc.collect()

    # Join df & prev_agg
    df = df.join(prev_agg, how='left', on='SK_ID_CURR')
    gc.collect()

    # Removal of non-useful variables
    del previous_appli, approved, approved_agg, refused, refused_agg, prev_agg
    gc.collect()

 # Import & traitement  data POS_CASH_balance
    pos_cash = pd.read_csv(os.path.join(
        folder_path, 'POS_CASH_balance.csv'), sep=',')

    # Categorical features with One-Hot encode
    pos_cash, pos_cash_cat = one_hot_encoder(pos_cash, True)
    gc.collect()

    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in pos_cash_cat:
        aggregations[cat] = ['mean']

    pos_agg = pos_cash.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    gc.collect()

    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos_cash.groupby('SK_ID_CURR').size()
    gc.collect()

    # Join df & pos_agg
    df = df.join(pos_agg, how='left', on='SK_ID_CURR')
    gc.collect()

    # Removal of non-useful variables
    del pos_cash, pos_agg
    gc.collect()

 # Import & traitement  data installments_payments
    installments_pay = pd.read_csv(os.path.join(
        folder_path, 'installments_payments.csv'), sep=',')
    
    # Categorical features with One-Hot encode
    installments_pay, installments_pay_cat = one_hot_encoder(installments_pay, True)
    gc.collect()

    # Percentage and difference paid in each installment (amount paid and installment value)
    installments_pay['PAYMENT_PERC'] = installments_pay['AMT_PAYMENT'] / installments_pay['AMT_INSTALMENT']
    installments_pay['PAYMENT_DIFF'] = installments_pay['AMT_INSTALMENT'] - installments_pay['AMT_PAYMENT']
    gc.collect()

    # Days past due and days before due (no negative values)
    installments_pay['DPD'] = installments_pay['DAYS_ENTRY_PAYMENT'] - installments_pay['DAYS_INSTALMENT']
    installments_pay['DBD'] = installments_pay['DAYS_INSTALMENT'] - installments_pay['DAYS_ENTRY_PAYMENT']
    installments_pay['DPD'] = installments_pay['DPD'].apply(lambda x: x if x > 0 else 0)
    installments_pay['DBD'] = installments_pay['DBD'].apply(lambda x: x if x > 0 else 0)
    gc.collect()

    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in installments_pay_cat:
        aggregations[cat] = ['mean']
    ins_agg = installments_pay.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    gc.collect()

    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = installments_pay.groupby('SK_ID_CURR').size()
    gc.collect()

    # Join df & ins_agg
    df = df.join(ins_agg, how='left', on='SK_ID_CURR')
    gc.collect()

    # Removal of non-useful variables
    del installments_pay, ins_agg

 # Import & traitement data credit_card_balance
    credit_card = pd.read_csv(os.path.join(
        folder_path, 'credit_card_balance.csv'),sep=',')
    gc.collect()

    # Categorical features with One-Hot encode
    credit_card, credit_card_cat = one_hot_encoder(credit_card, True)
    gc.collect()

    # General aggregations
    credit_card.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = credit_card.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    gc.collect()

    # Count credit card lines
    cc_agg['CC_COUNT'] = credit_card.groupby('SK_ID_CURR').size()
    gc.collect()

    # Join df & cc_agg
    df = df.join(cc_agg, how='left', on='SK_ID_CURR')
    gc.collect()

    # Removal of non-useful variables
    del credit_card, cc_agg
    
    # Rename columns w/o special characters
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    gc.collect() 
    
 # return application file consolidated
    return df
gc.collect()


# Non significative columns, N/A & Inf traitement
def clean_data(df):
    # Determine the bolean type variables and the non-significant variables
    nunique_app = df.nunique().reset_index()
    nunique_app = nunique_app[nunique_app[0] == 1]
    no_signi_value = nunique_app['index'].tolist()
    nunique_app = df.nunique().reset_index()
    nunique_app = nunique_app[nunique_app[0] == 2]
    bool_values = nunique_app['index'].tolist()
    del nunique_app
    gc.collect()

    # Removal of columns of non-significant variables
    df = df.drop(columns=no_signi_value)
    del no_signi_value
    gc.collect()

    # List of quantitative variables
    quanti =df.select_dtypes(include = ['float64']).columns.tolist()
    gc.collect()

    # Infiny value search (positive)
    infiny_pos = pd.DataFrame(df[quanti].max().sort_values()).reset_index()
    infiny_pos = infiny_pos[infiny_pos[0]==np.inf]
    # List of columns containing infinite values
    inf_columns = infiny_pos["index"].tolist()
    # Replace Infiny values
    df[inf_columns] = df[inf_columns].replace(np.inf, df[df[inf_columns]!=np.inf].max())
    del inf_columns, infiny_pos
    gc.collect()

    # Infiny value search (negative)
    infiny_neg = pd.DataFrame(df[quanti].min().sort_values()).reset_index()
    infiny_neg = infiny_neg[infiny_neg[0]==np.inf]
    # List of columns containing infinite values
    inf_columns = infiny_neg["index"].tolist()
    # Replace Infiny values
    df[inf_columns] = df[inf_columns].replace(np.inf, df[df[inf_columns]!=np.inf].min())
    del inf_columns, infiny_neg
    gc.collect()

    # Columns name list
    columns = df.columns.to_list()
    gc.collect()

    # List of columns containing na value
    na_columns = []
    for c in columns:
        nan_row = df[c].isna().values.any()
        if nan_row == True:
            na_columns.append(c)
    del columns, c, nan_row

    # Replace na values
    simple_impute = SimpleImputer(missing_values=np.nan, strategy='median').fit(df[na_columns])
    #df = df[columns].apply(lambda x:x.fillna(x.median()))
    df[na_columns]=simple_impute.transform(df[na_columns])
    del na_columns
    gc.collect()

    # Transform boolean values in integer
    df[bool_values] = df[bool_values].astype('int')
    del bool_values
    gc.collect()

    # Return df file clean
    return df
gc.collect()


# Normalize numerical columns
def norm_data(df):


    # List of quantitatives variables
    quanti = df.select_dtypes(include = ['float64']).columns.tolist()
    # Normalization of quantitatives variables
    df[quanti] = normalize(df[quanti])
    gc.collect()
    
    # Return df file normalized
    return df
gc.collect()
