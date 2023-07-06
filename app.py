# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 14:01:50 2023

@author: Abayo
"""
# loading required libraries
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler
import pickle
import json

#loading an instance of fastapi

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
    
    )

# mentioning the input formats we need

class modelTemp_input(BaseModel):
    body_temperature: float    
    
# loading the saved models
bt_model = pickle.load(open('modelSVR_BT.0.1.0.sav','rb'))

# Creating an APIs

@app.post('/body_temperature_prediction')

def body_temperature_pred(input_parameters: modelTemp_input):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    temp = input_dictionary['body_temperature']

    input_list1 = [temp]
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    predictionBT = scaler.inverse_transform(bt_model.predict(scaler.transform([[input_list1]])).reshape(-1,1))
    
    return predictionBT
    
