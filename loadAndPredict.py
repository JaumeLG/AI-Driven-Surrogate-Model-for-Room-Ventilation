import os
from decimal import Decimal
import numpy as np
from numpy import genfromtxt
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import pandas as pd

import keras
from keras.models import load_model
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Flatten
from keras.layers import Input, Concatenate, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping

import pickle

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


#Preamble
input_path="./inputs/"
output_path="./outputs/"

database_input=[]
input_all=[]
database_output=[]
total_samples_in=0
total_samples_out=0

#Loading inputs
for filename in sorted(glob.glob("inputs/initialDataNN_HR_T_case_*.csv")):
    case=int(filename.split("_")[5].split(".")[0])
    
    rawData=pd.read_csv(filename, delimiter=',')
    rawData.drop(rawData.columns[[0,1,2]],axis=1,inplace=True)
    rawData["HR"]*=100
    rawData["T"]-=273.15
    
    refinedData=rawData.to_numpy()
    refinedData=refinedData.flatten()

    
    with open ("./cases_list_humidity.txt", "r") as file:
        lines=file.readlines()
        caudalDeshum=np.array(lines[case+1].split(" ")[2],dtype=float,ndmin=1)
        caudalCalle=np.array(lines[case+1].split(" ")[4],dtype=float,ndmin=1)
        caudalExtrac=np.array(lines[case+1].split(" ")[6],dtype=float,ndmin=1)
        Tdeshum=np.array(lines[case+1].split(" ")[12],dtype=float,ndmin=1)-273.15
        Tcalle=np.array(lines[case+1].split(" ")[14],dtype=float,ndmin=1)-273.15
        HRdeshum=np.array(lines[case+1].split(" ")[22],dtype=float,ndmin=1)
        HRcalle=np.array(lines[case+1].split(" ")[24],dtype=float,ndmin=1)
        
    refinedData=np.concatenate((refinedData,HRdeshum,Tdeshum,HRcalle,Tcalle,caudalDeshum,caudalCalle,caudalExtrac),axis=0)
    
    
    input_all.append(refinedData)

    total_samples_in += 1

#Loading outputs

for filename in sorted(glob.glob("outputs/dataNN_HR_T_case_*.csv")):
    
    rawData=pd.read_csv(filename, delimiter=',')
    rawData.drop(rawData.columns[[0,1,2]],axis=1,inplace=True)
    rawData["HR"]*=100
    rawData["HR"]=rawData["HR"].clip(lower=None,upper=100)
    rawData["T"]-=273.15
    
    refinedData=rawData.to_numpy()
    refinedData=refinedData.flatten()
    
    database_output.append(refinedData)

    total_samples_out += 1

#Prepare inputs and process data

input_all=np.array(input_all,dtype=float)
database_output=np.array(database_output,dtype=float)

sensor2 = 12
sensor3 = 1
sensor4 = 21     

cols=np.array([0,1,sensor2*2,sensor2*2+1,sensor3*2,sensor3*2+1,sensor4*2,sensor4*2+1,44,45,46,47,48,49,50])
database_input=input_all[:,cols]


scalerX=StandardScaler()
scalerY=StandardScaler()
X=scalerX.fit_transform(database_input)  
Y=scalerY.fit_transform(database_output)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=37)

n_batch = 24

#Loading model
model=load_model('./NN_Oceanografic.h5',compile=False)

#Generating predictions
preds=model.predict(X_test)

#Dividing preds between Relative Humidity and Temperature

RH_truth=Y_test[:,0::2]
T_truth=Y_test[:,1::2]
RH_pred=pred[:,0::2]
T_pred=pred[:,1::2]








