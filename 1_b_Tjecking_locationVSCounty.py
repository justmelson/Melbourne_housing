# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 23:53:18 2021

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


data = pd.read_pickle("./pickle/train_set.pkl")


#%% Plot Lat Lon
ax = data.plot(kind='scatter',x='Lattitude',y='Longtitude',color='white')


#%% Get all Area codes
Area_codes=data['CouncilArea'].value_counts().index.to_numpy()

Area_codes=Area_codes[1:29 | 31]
# Set index of dataset to council area
data.set_index("CouncilArea", inplace=True, drop=False)


## Plot all area Codes
i=0
for Area in Area_codes:
    #Area=Area_codes[-3]
    Area_dat=data.loc[Area]
    r=(1/15*i+0.2) *(i <= 10 )
    g=(1/15*i-0.4) *( i>10 and i<=20 ) 
    b=(1/15*i-1) *( i>20 and i<=30 )
    plt.figure(2)
    Area_dat.plot(kind='scatter',x='Longtitude',y='Lattitude',color=[(r, g, b)],s=1.5,ax=ax)
    i+=1

ax.set_ylim(-38.2,-37.5)
ax.set_xlim(144.4,145.5)
fig = ax.get_figure()
fig.savefig("./figures/O2_5_Council_scatter.eps",bbox_inches = "tight")
#&& All data
#ax1 = data.plot(kind='scatter',x='Lattitude',y='Longtitude',color='red')

#%% No area
No_area=data.loc[ data['CouncilArea'].isnull() ]

No_area.plot(kind='scatter',y='Lattitude',x='Longtitude',color='black',s=0.5, ax=ax)
fig.savefig("./figures/O2_5_Council_scatter_Lost.eps",bbox_inches = "tight")
plt.show()

