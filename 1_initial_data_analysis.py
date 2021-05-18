# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 11:11:27 2021

"""

#%% IMPORTING PACKAGES
import pandas as pd
import matplotlib.pyplot as plt
import os


#%% IMPORTING DATA FROM CSV
# current working directory
dirname = os.path.dirname(__file__)

# path to the data
datadir = os.path.join(dirname, 'data/melb_data.csv')

# loading data to a pandas dataframe
data = pd.read_csv(datadir)


#%% INITIAL DATA ANALYSIS
# general info about the dataset
data.info()

# the few propperties with no data for car parking sports are removed
data.dropna(subset=['Car'],inplace=True)

# why aren't no. of bedrooms and bathrooms integers?
print('\n')
print(data['Bedroom2'].value_counts())
print('\n')
print(data['Bathroom'].value_counts())
print('\n')
print(data['Car'].value_counts())
# converting them to integers
convert_dict = {'Bedroom2': int, 'Bathroom': int, 'Car': int}
data = data.astype(convert_dict)

# general info about the updated dataset
print('\n')
data.info()


# checking the number of catagories in 'Suburb', 'Type', 'Method', 'SellerG', 
# 'CouncilArea', and 'Regionname'
print('\n')
print(data['Suburb'].value_counts())
print('\n')
print(data['Type'].value_counts())
print('\n')
print(data['Method'].value_counts())
print('\n')
print(data['SellerG'].value_counts())
print('\n')
print(data['CouncilArea'].value_counts())
print('\n')
print(data['Regionname'].value_counts())
# Type, Method and Regionname are definitely categorical


# determining statistical properties of the numerical features
statistics = data.describe()

# plotting histograms for the numerical features
data.hist(bins=50, figsize=(20,15))
plt.savefig("./figures/O2_4_histograms.eps",bbox_inches = "tight")
plt.close()


# discrete values for number of rooms
print('\n')
print(data['Rooms'].value_counts())
# discrete number of property counts
print('\n')
print(data['Propertycount'].value_counts())
# discrete number of postcodes
print('\n')
print(data['Postcode'].value_counts())


# removing rows with year of construction before 1793
removedRows = (data['YearBuilt']>=1793)

# plotting histogram for year built without erronous values
histogram_data = data.loc[removedRows]
histogram_data.hist(bins=50, figsize=(10,7),column='YearBuilt')
plt.savefig("./figures/O2_4_histogram_YearBuilt.eps",bbox_inches = "tight")
plt.close()


# finding the max percentile of 'Landsize' and 'BuildingArea' to keep
maxPercentile = 0.995
Landsize_maxPercentile = data['Landsize'].quantile(maxPercentile)
BuildingArea_maxPercentile = data['BuildingArea'].quantile(maxPercentile)

# removing rows with values above the maximum percentile
removedRows = (data['Landsize']<=Landsize_maxPercentile) & \
    (data['BuildingArea']<=BuildingArea_maxPercentile)

# plotting histograms for land size and building area without outliers
histogram_data = data.loc[removedRows]
histogram_data.hist(bins=50, figsize=(20,7),column=['Landsize','BuildingArea'])
plt.savefig("./figures/O2_4_histograms_noOutliers.eps",bbox_inches = "tight")
plt.close()


#%% PICKLING THE DATA
data.to_pickle("./pickle/data.pkl")
