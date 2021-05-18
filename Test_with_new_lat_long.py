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
Onehot=['Type','Method']
#
# Columns which are to be dropped

Drop=['Address','SellerG','Regionname','Suburb','Postcode','CouncilArea']
      #,'Distance','Postcode','Bedroom2','Bathroom','Propertycount','Car','YearBuilt','Date',]
#'Address','Suburb','Type','Method','SellerG','Regionname'
# Example of the Plot function
#MDL.Plot_Long_lat(MDL.X_train,'CouncilArea')
#MDL.Plot_Long_lat(MDL.X_train,'Regionname')

#%%
# Preprocess Dataframe
MDL.X_train_processed,MDL.y_train_processed=MDL.Preprocess(MDL.X_train,MDL.y_train,Onehot,Drop,K=3,Plot=True,Edist=True)

#MDL.X_train_processed.drop("Lattitude", axis='columns', inplace=True)
#MDL.X_train_processed.drop("Longtitude", axis='columns', inplace=True)

# Husk kneighbors predict
# Make dataframe into PURE np array
X_train,Header_X=MDL.Dataframe_to_np(MDL.X_train_processed)

y_train,Header_y=MDL.Dataframe_to_np(MDL.y_train_processed)

y_train=np.ravel(y_train)


#%%
y_train_mean=np.mean(y_train)
y_train_std=np.std(y_train)
print(f'Expected mean = {y_train_mean} \nExpected std {y_train_std}\n')
#%% SKLEARN TIME



# SGD regressor
y =y_train
X = X_train
# Always scale the input. The most convenient way is to use a pipeline.

SGD_reg = make_pipeline(StandardScaler(),
                     SGDRegressor(max_iter=3000, tol=1e-4))



scores = cross_val_score(SGD_reg,X,y,
                         scoring="r2",cv=5)

print('Score for SGD')
display_scores(scores)


SGD_reg.fit(X_train,y_train)
y_pred_SGD=SGD_reg.predict(X_train)

print(f'Expected mean = {np.mean(y_pred_SGD)} \nExpected std {np.std(y_pred_SGD)}\n')


#%% Forrest regressor


forest_reg= RandomForestRegressor()


# Cross validated Tree regressor

scores = cross_val_score(forest_reg,X_train,y_train,
                         scoring="r2",cv=5)


from sklearn.model_selection import cross_val_predict
Predictions=cross_val_predict(forest_reg,X_train,y_train,
                         cv=5)

Error=y_train-Predictions
plt.hist(Error,bins=30)
plt.title('Error histogram')
plt.xlabel('Price')
plt.ylabel('Count')
plt.savefig("./figures/Error_hist.eps",bbox_inches = "tight")
print(f'Mean {np.mean(Error)}' )
print(f'Sigma {np.std(Error)}' )
print('Score for random forest')
display_scores(scores)



forest_reg.fit(X_train,y_train)
y_pred_forest=forest_reg.predict(X_train)

print(f'Expected mean = {np.mean(y_pred_forest)} \nExpected std {np.std(y_pred_forest)}\n')
#

