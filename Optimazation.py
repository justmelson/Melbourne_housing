# -*- coding: utf-8 -*-
"""
Created on Tue May 11 13:00:16 2021

"""

#%% IMPORTING PACKAGES

# Different stuff
from datetime import datetime
from dateutil.parser import parse
from time import time


# Standard things
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Own modules
from Pipeline import Pipeline


# Regressors
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

# Pipelines
from sklearn.pipeline import make_pipeline


# Metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

# Scalers
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Optimization
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from scipy.stats import randint

def display_scores(scores):
        print("Scores",scores)
        print("Mean:",scores.mean())
        print("STD:",scores.std())

#%% Set random seed
# Appently sklearn uses numpys seed so it can be set here :p
np.random.seed(31415)        
#%% IMPORTING DATA FROM PICKLE
Train_set = pd.read_pickle("./pickle/train_set.pkl")
Test_set  = pd.read_pickle("./pickle/test_set.pkl")

# Initialize Pipeline with all data
MDL=Pipeline(Train_set,Test_set)




#%% Preprocessing Traning data

# Coloumn names which are to be made into one hot parameters
Onehot=['Type','Method','CouncilArea']
#
# Columns which are to be dropped

Drop=['Address','SellerG','Regionname','Suburb','Postcode','BuildingArea','YearBuilt']
      #,'Distance','Postcode','Bedroom2','Bathroom','Propertycount','Car','YearBuilt','Date',]
#'Address','Suburb','Type','Method','SellerG','Regionname'
# Example of the Plot function
#MDL.Plot_Long_lat(MDL.X_train,'CouncilArea')
#MDL.Plot_Long_lat(MDL.X_train,'Regionname')

#%%
# Preprocess Dataframe
MDL.X_train_processed,MDL.y_train_processed=MDL.Preprocess(MDL.X_train,MDL.y_train,Onehot,Drop,K=3,Plot=False)

# Husk kneighbors predict

# Make dataframe into PURE np array
X_train,Header_X=MDL.Dataframe_to_np(MDL.X_train_processed)

y_train,Header_y=MDL.Dataframe_to_np(MDL.y_train_processed)

y_train=np.ravel(y_train)




#%% Make pipeline for optimazation
RanSearch_pipe = make_pipeline(
                               RandomForestRegressor(n_jobs=-1) )

#%% Get Params
Names= RanSearch_pipe.get_params().keys()
print(Names)
#%%
print('Starting CV')
distributions = dict(
                     randomforestregressor__n_estimators=randint(10,300),
                     randomforestregressor__ccp_alpha= uniform(0.00005,0.005))

Random_search = RandomizedSearchCV(RanSearch_pipe, distributions,
                                   n_iter=400,
                                   verbose=10,
                                   cv=3)

search = Random_search.fit(X_train, y_train)

search.best_params_

print('Ok')


#%% Finish sound
import winsound
duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)