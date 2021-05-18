# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 14:55:41 2021
"""

#%% IMPORTING PACKAGES
import pandas as pd


#%% IMPORTING DATA FROM PICKLE
data = pd.read_pickle("./pickle/data.pkl")


#%% SPLITTING THE DATA INTO TRAIN AND TEST SETS
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)


#%% PICKLING THE SETS
train_set.to_pickle("./pickle/train_set.pkl")
test_set.to_pickle("./pickle/test_set.pkl")