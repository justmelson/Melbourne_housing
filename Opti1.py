
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
Onehot=['Type','Method','CouncilArea']
#
# Columns which are to be dropped

Drop=['Address','SellerG','Regionname','Suburb','Postcode','YearBuilt','BuildingArea']
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


forest_reg= RandomForestRegressor(n_estimators=200)


# Cross validated Tree regressor

scores = cross_val_score(forest_reg,X_train,y_train,
                         scoring="r2",cv=5)

print('Score for random forest')
display_scores(scores)



forest_reg.fit(X_train,y_train)
y_pred_forest=forest_reg.predict(X_train)

print(f'Expected mean = {np.mean(y_pred_forest)} \nExpected std {np.std(y_pred_forest)}\n')
#

#%% Neural network
 
# # https://scikit-learn.org/stable/auto_examples/inspection/plot_partial_dependence.html#sphx-glr-auto-examples-inspection-plot-partial-dependence-py
# # F??lger ovenst??ende eksempel


print("Training MLPRegressor...")
tic = time()
est = make_pipeline(QuantileTransformer(),
                     MLPRegressor(hidden_layer_sizes=(300, 300,300),
                                  learning_rate_init=0.001,
                                  early_stopping=True,
                                  alpha=0.0002))

scores = cross_val_score(est,X_train,y_train,
                         scoring="r2",cv=5)

print('Score for MLP')
display_scores(scores)

est.fit(X_train,y_train)
y_pred_MLP=est.predict(X_train)

print(f'Expected mean = {np.mean(y_pred_MLP)} \nExpected std {np.std(y_pred_MLP)}\n')



#%%

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor


clf = HistGradientBoostingRegressor(max_iter=100).fit(X_train, y_train)
#clf.score(X_test, y_test)

scores = cross_val_score(est,X_train,y_train,
                         scoring="r2",cv=5)


display_scores(scores)
#