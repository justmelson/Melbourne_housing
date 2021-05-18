# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 20:23:48 2021

"""
#%% IMPORTING PACKAGES
import pandas as pd
from Pipeline import Pipeline
import numpy as np

from datetime import datetime
from dateutil.parser import parse


import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
a=2


def display_scores(scores):
        print("Scores",scores)
        print("Mean:",scores.mean())
        print("STD:",scores.std())
        
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

Drop=['Address','SellerG','Regionname','Suburb','Postcode','CouncilArea',]
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
Scaler=StandardScaler()

X_train=Scaler.fit_transform(X_train)
#%% SKLEARN TIME



# SGD regressor
y =y_train
X = X_train
# Always scale the input. The most convenient way is to use a pipeline.
reg = SGDRegressor(max_iter=3000, tol=1e-4)
reg.fit(X, y)

y_pred=reg.predict(X_train)



SGD_mse= mean_squared_error(y,y_pred)

SGD_rmse=np.sqrt(SGD_mse)

print(f' RMSE is {SGD_rmse}$')


#%% Tree regressor
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()

tree_reg.fit(X_train,y_train)

y_pred=tree_reg.predict(X_train)

Tree_mse= mean_squared_error(y,y_pred)

Tree_rmse=np.sqrt(SGD_mse)

print(f' RMSE is {SGD_rmse}$')

#%% Cross validated Tree regressor
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg,X_train,y_train,
                         scoring="r2",cv=5)


display_scores(scores)
def display_scores(scores):
        print("Scores",scores)
        print("Mean:",scores.mean())
        print("STD:",scores.std())

print('Det er er SUUUPER dårligt')

#%% Forrest regressor

from sklearn.ensemble import RandomForestRegressor
forest_reg= RandomForestRegressor()
forest_reg.fit(X_train,y_train)


#%% Cross validated Tree regressor

scores = cross_val_score(forest_reg,X_train,y_train,
                         scoring="r2",cv=5)


display_scores(scores)

print('Det er er SUUUPER dårligt')


#%% PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
X_train_PCA = pca.fit_transform(X_train)

forest_reg_PCA= RandomForestRegressor()
forest_reg_PCA.fit(X_train_PCA,y_train)

y_pred=forest_reg_PCA.predict(X_train_PCA)

Error=(y_train-y_pred)/y_train*100
forest_mse= mean_squared_error(y,y_pred)

forest_rmse=np.sqrt(forest_mse)

print(f' RMSE is {forest_rmse}$')

print(f'Percentage error {np.std(Error)}')


#%% Cross validated Tree regressor

scores = cross_val_score(forest_reg_PCA,X_train_PCA,y_train,
                         scoring="r2",cv=5)


display_scores(scores)

print('Det er er SUUUPER dårligt')


#%% Neural network
 
# # https://scikit-learn.org/stable/auto_examples/inspection/plot_partial_dependence.html#sphx-glr-auto-examples-inspection-plot-partial-dependence-py
# # Følger ovenstående eksempel
from time import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.neural_network import MLPRegressor

print("Training MLPRegressor...")
tic = time()
est = make_pipeline(QuantileTransformer(),
                     MLPRegressor(hidden_layer_sizes=(100, 100),
                                  learning_rate_init=0.1,
                                  early_stopping=True))
est.fit(X_train, y)
y_pred=est.predict(X_train)
print(f"done in {time() - tic:.3f}s")
print(f"Test R2 score: {est.score(X_train, y):.2f}")
print(f"Test MSE score: {mean_squared_error(y_train, y_pred):.2f}")
print(f"Test RMSE score: {np.sqrt( mean_squared_error(y_train, y_pred) ):.2f}")


#%% PCA
print("Training MLPRegressor...")
tic = time()
est = make_pipeline(QuantileTransformer(),
                     MLPRegressor(hidden_layer_sizes=(300, 300,300),
                                  learning_rate_init=0.051,
                                  early_stopping=True))
est.fit(X_train_PCA, y_train)
y_pred=est.predict(X_train_PCA)
print(f"done in {time() - tic:.3f}s")
print(f"Test R2 score: {est.score(X_train_PCA, y):.2f}")
print(f"Test MSE score: {mean_squared_error(y_train, y_pred):.2f}")
print(f"Test RMSE score: {np.sqrt( mean_squared_error(y_train, y_pred) ):.2f}")



scores = cross_val_score(est,X_train_PCA,y_train,
                         scoring="r2",cv=5)


display_scores(scores)


print('Det er er SUUUPER dårligt')
# #%% PCA transform
# from sklearn.decomposition import PCA

# pca = PCA()
# X2D = pca.fit_transform(X_train)

# #%% Make som description
# print( pca.explained_variance_ratio_ )

# cumsum = np.cumsum ( pca.explained_variance_ratio_ )
# size_c=np.size(cumsum)
# plt.plot(list(range(size_c)), cumsum)

# #%% 15 variables
# pca95 = PCA(n_components=200)
# X2D95 = pca95.fit_transform(X_train)

# print("Training MLPRegressor...")
# tic = time()
# est = make_pipeline(QuantileTransformer(),
#                     MLPRegressor(hidden_layer_sizes=(100, 100),
#                                  learning_rate_init=0.05,
#                                  early_stopping=True))
# est.fit(X2D95, y_train)
# y_pred=est.predict(X2D95)
# Error=y_pred-np.ravel(y_train)
# print(f"done in {time() - tic:.3f}s")
# print(f"Test R2 score: {est.score(X2D95, y_train):.2f}")


#%% Test qunatile
quan=QuantileTransformer()

New=quan.fit_transform(X_train)


