
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 12:28:00 2021

"""
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 20:23:48 2021

@author: anton
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

Drop=['Address','SellerG','Regionname','Suburb','Postcode']
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

forest_reg= RandomForestRegressor(n_estimators=20)



print("Training MLPRegressor...")
tic = time()
est = make_pipeline(QuantileTransformer(),
                     MLPRegressor(hidden_layer_sizes=(1000, 1000,1000,1000),
                                  learning_rate_init=0.001,
                                  early_stopping=True,
                                  alpha=0.0005))

scores = cross_val_score(est,X_train,y_train,
                         scoring="r2",cv=5)

print('Score for MLP')
display_scores(scores)

est.fit(X_train,y_train)
y_pred_MLP=est.predict(X_train)

print(f'Expected mean = {np.mean(y_pred_MLP)} \nExpected std {np.std(y_pred_MLP)}\n')
#