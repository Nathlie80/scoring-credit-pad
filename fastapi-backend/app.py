# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 15:42:35 2022

@author: nathl
"""

# 1. Library imports
from fastapi import FastAPI
from joblib import load
import numpy as np
import pandas as pd
from pydantic import BaseModel
import uvicorn

# 2. Create app and model objects

app = FastAPI()

## import model
model = load('lgbm_model.joblib')

## threshold calculate for this model
threshold = 0.18

## Import data
application = pd.read_pickle('sample_norm.pkl')


## List of columns to apply model
list_columns = application.columns.to_list()
list_columns.remove('SK_ID_CURR')
list_columns.remove('TARGET')

## class client creation
class Client(BaseModel):
    id : int
    
@app.get("/")
def hello_world():
    return {"msg": "Client risk probability API"}


# 3. Expose the prediction functionality, make a prediction from  data 
## and return the predicted and probability

@app.post('/predict')
def pred_credit(client : Client):
    input_data = client.dict()
    id = int(input_data["id"])
    # data_client = application.loc[application['SK_ID_CURR'] == id, list_columns]
    data_client = application[application["SK_ID_CURR"]== id]
    probability = model.predict_proba(data_client[list_columns])
    prediction = np.where(probability[:,1] < threshold, 0, 1)
    
    return {'probability' : probability.tolist(), 'prediction' : prediction.tolist()}

# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__app__':
    uvicorn.run(app, host='127.0.0.1', port=8000)