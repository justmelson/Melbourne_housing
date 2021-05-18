
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
from sklearn.metrics import r2_score

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
MDL.X_train_processed,MDL.y_train_processed=MDL.Preprocess(MDL.X_train,MDL.y_train,Onehot,Drop,K=3,Plot=True)

# Husk kneighbors predict

# Make dataframe into PURE np array
X_train,Header_X=MDL.Dataframe_to_np(MDL.X_train_processed)

y_train,Header_y=MDL.Dataframe_to_np(MDL.y_train_processed)

y_train=np.ravel(y_train)


#%%
y_train_mean=np.mean(y_train)
y_train_std=np.std(y_train)
print(f'Expected mean = {y_train_mean} \nExpected std {y_train_std}\n')

#%% Forrest regressor


forest_reg= RandomForestRegressor()


# Cross validated Tree regressor

scores = cross_val_score(forest_reg,X_train,y_train,
                         scoring="r2",cv=5)

print('Score for random forest')
display_scores(scores)



forest_reg.fit(X_train,y_train)
y_pred_forest=forest_reg.predict(X_train)

print(f'Expected mean = {np.mean(y_pred_forest)} \nExpected std {np.std(y_pred_forest)}\n')
#

#%% Process the test set

Onehot=['Type','Method','CouncilArea']
#
# Columns which are to be dropped

Drop=['Address','SellerG','Regionname','Suburb','Postcode']

MDL.X_test_processed,MDL.y_test_processed=MDL.Preprocess(MDL.X_test,MDL.y_test,Onehot,Drop,K=3,Plot=False)


Np_array=np.zeros((1409,1))


X_test,Header_Xtest=MDL.Dataframe_to_np(MDL.X_test_processed)

y_test,Header_ytest=MDL.Dataframe_to_np(MDL.y_test_processed)

y_test=np.ravel(y_test)



#%% Insert row for two missing country areas
Row1=23 # Row before cardinia
Row2=34 # Row before Manningham

X_test=np.insert(X_test, Row1, np.zeros(1409),axis=1)
X_test=np.insert(X_test, Row2, np.zeros(1409),axis=1)
y_pred_test=forest_reg.predict(X_test)

R2_test=r2_score(y_test, y_pred_test)
print(R2_test)
