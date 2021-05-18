
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

Drop=['Address','SellerG','Suburb','Postcode']
      #,'Distance','Postcode','Bedroom2','Bathroom','Propertycount','Car','YearBuilt','Date',]
#'Address','Suburb','Type','Method','SellerG','Regionname'
# Example of the Plot function
#MDL.Plot_Long_lat(MDL.X_train,'CouncilArea')
#MDL.Plot_Long_lat(MDL.X_train,'Regionname')

#%%
# Preprocess Dataframe
MDL.X_train_processed,MDL.y_train_processed=MDL.Preprocess(MDL.X_train,MDL.y_train,Onehot,Drop,K=3,Plot=False)

# Husk kneighbors predict

y=MDL.X_train.Lattitude

x=MDL.X_train.Longtitude

Z=MDL.y_train.Price

ax = plt.axes(projection='3d')
#plt.plot(x,y)
ax.scatter3D(x, y, Z, c=Z,s=1);
plt.title('Price based on location')
plt.xlabel('Longtitude')
plt.ylabel('Lattitude')
plt.clabel('Price')
plt.rcParams['xtick.labelsize'] = 5
#ax.view_init(elev=90., azim=0)
plt.savefig("./figures/3d_Price.eps",bbox_inches = "tight")
#%% Tjeck based on middle


y_mean=np.mean(y)
x_mean=np.mean(x)
R=np.sqrt( (x-x_mean)**2 + (y-y_mean)**2)

MDL.Plot_Long_lat(MDL.X_train,'Regionname')
#

plt.scatter(x_mean,y_mean,s=10)
plt.title('Mean location')

#%% Calculate cartesian distance from geometric center
Lat_mean_rad=y_mean*3.14/180
Long_mean_rad=x_mean*3.14/180
Lat_rad=y*3.14/180

Long_rad=x*3.14/180

#%% Calc haversine
dLat=Lat_rad-Lat_mean_rad
dlong=Long_rad-Long_mean_rad
a =np.sin(dLat/2)**2 + np.abs(np.cos(Lat_rad)*np.cos(Long_rad)*np.sin(dlong/2)**2 )

c=2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
d=6371e3*c

#plt.figure()
plt.scatter(R,Z,s=0.2)
plt.xlabel('Euclidian distance')
plt.ylabel('Price')
plt.title('Price versus Euclidian distance of Lat/Long in Deg')
plt.savefig("./figures/Correlation_C_Price.eps",bbox_inches = "tight")
