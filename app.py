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
class modelHeartRate_input(BaseModel):
    heart_rate: float
class modelRespiratoryRate_input(BaseModel):
    respiratory_rate: float
class modelSPO2_input(BaseModel):
    SPO2: float
class modelSystolicPressure_input(BaseModel):
    systolic_blood_pressure: float
class modelDiastolicPressure_input(BaseModel):
    diastolic_blood_pressure: float
    
    
# loading the saved models
bt_model = pickle.load(open('modelSVR_BT.0.1.0.sav','rb'))
hr_model = pickle.load(open('modelSVR_HR.0.1.0.sav','rb'))
rr_model = pickle.load(open('modelSVR_RR.0.1.0.sav','rb'))
spo2_model = pickle.load(open('modelSVR_SPO2.0.1.0.sav','rb'))
sys_model = pickle.load(open('modelSVR_SBP.0.1.0.sav','rb'))
dys_model = pickle.load(open('modelSVR_DBP.0.1.0.sav','rb'))

# Creating an APIs

@app.post('/body_temperature_prediction')

def body_temperature_pred(input_parameters: modelTemp_input):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    temp = input_dictionary['body_temperature']

    input_list1 = [temp]
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    predictionBT = scaler.inverse_transform(bt_model.predict(scaler.transform([[input_list1]])).reshape(-1,1))
    
    return print(predictionBT)
    

@app.post('/heart_rate_prediction')

def heart_rate_pred(input_parameters: modelHeartRate_input):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    hr = input_dictionary['heart_rate']

    input_list2 = [hr]
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    predictionHR = scaler.inverse_transform(hr_model.predict(scaler.transform([[input_list2]])).reshape(-1,1))
    
    return print(predictionHR)

@app.post('/respiratory_rate_prediction')

def respiratory_rate_pred(input_parameters: modelRespiratoryRate_input):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    rr = input_dictionary['respiratory_rate']

    input_list3 = [rr]
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    predictionRR = scaler.inverse_transform(hr_model.predict(scaler.transform([[input_list3]])).reshape(-1,1))
    
    return print(predictionRR)

@app.post('/spo2_prediction')

def spo2_pred(input_parameters: modelSPO2_input):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    spo2 = input_dictionary['spo2']

    input_list4 = [spo2]
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    predictionSPO2 = scaler.inverse_transform(hr_model.predict(scaler.transform([[input_list4]])).reshape(-1,1))
    
    return print(predictionSPO2)

@app.post('/systolic_prediction')

def sys_pred(input_parameters: modelSystolicPressure_input):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    sys = input_dictionary['systolic_blood_pressure']

    input_list5 = [sys]
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    predictionSYS = scaler.inverse_transform(hr_model.predict(scaler.transform([[input_list5]])).reshape(-1,1))
    
    return print(predictionSYS)

@app.post('/diastolic_prediction')

def dys_pred(input_parameters: modelDiastolicPressure_input):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    dias = input_dictionary['diastolic_blood_pressure']

    input_list6 = [dias]
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    predictionDIA = scaler.inverse_transform(hr_model.predict(scaler.transform([[input_list6]])).reshape(-1,1))
    
    return print(predictionDIA)
