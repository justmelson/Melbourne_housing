# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 20:23:35 2021

"""

class Pipeline:
    def __init__(self,Train_set,Test_set):
        # Initialize. Mostly saves the test and training set into X and y
        self.y_train = Train_set[['Price']]
        self.y_test = Test_set[['Price']]
        self.X_train = Train_set
        self.X_test = Test_set
        
        self.X_train.drop('Price',axis='columns', inplace=True)
        self.X_test.drop('Price',axis='columns', inplace=True)
        
        
    def Plot_Long_lat(self,data,Category):
        # Plot function. Plots the lat/long in council areas NaN data is black
        import matplotlib.pyplot as plt
        #%% Plot Lat Lon
        ax = data.plot(kind='scatter',x='Lattitude',y='Longtitude',color='white') ;
        #%% Get all Area codes
        Area_codes=data[Category].value_counts().index.to_numpy()

        Area_codes=Area_codes[0:31];
        # Set index of dataset to council area
        data=data.set_index(Category, inplace=False, drop=False);
        
        
        ## Plot all area Codes
        i=0
        for Area in Area_codes:
            #Area=Area_codes[-3]
            Area_dat=data.loc[Area] ;
            r=(1/15*i+0.1) *(i <= 10 );
            g=(1/15*i-0.5) *( i>10 and i<=20 ) ;
            b=(1/15*i-1) *( i>20 and i<=30 );

            plt.figure(2) ;
            Area_dat.plot(kind='scatter',x='Longtitude',y='Lattitude',color=[(r, g, b)],s=1.5,ax=ax) ;
            i+=1 ;
        
        ax.set_ylim(-38.2,-37.5) ;
        ax.set_xlim(144.4,145.5) ;
        fig = ax.get_figure() ;
        #fig.savefig("./figures/O2_5_Council_scatter.eps",bbox_inches = "tight")
        #&& All data
        #%% No area
        No_area=data.loc[ data[Category].isnull() ] ;
        No_area.plot(kind='scatter',y='Lattitude',x='Longtitude',color='black',s=0.5, ax=ax) ;
        print('Plottet lat/long OK')
    def K_neighbor_Areacodes(self,Dataset,K=3):
        #Tjecks if area codes are full and puts them closets to nearest neighbor
        
        from sklearn.neighbors import KNeighborsClassifier
        import numpy as np

        # Split into train,test
        X_train=Dataset[['Lattitude','Longtitude']][(Dataset['CouncilArea'].isnull())==False ]
        X_test=Dataset[['Lattitude','Longtitude']][(Dataset['CouncilArea'].isnull())==True ]
        y_label = Dataset['CouncilArea'][(Dataset['CouncilArea'].isnull())==False]

        
        K_model= KNeighborsClassifier(n_neighbors=K)
        K_model.fit(X_train, y_label)

        y_pred=K_model.predict(X_test)

        Empty_index=np.where(Dataset['CouncilArea'].isnull())[0]

        
        Dataset.loc[Dataset.CouncilArea.isnull(), 'CouncilArea'] = y_pred
        print('Fitted council area OK')
        
        return Dataset
    def Drop_NaN_rows(self,Dataset,Labels,Row_Col='index'):
        # Drop Nan rows
        import pandas as pd
        Row_before=Dataset['Rooms'].count()
        
        Dataset=pd.concat([Dataset,Labels],axis=1) # Appedn labels
        Dataset=Dataset.dropna(inplace=False)
        
        Labels=Dataset[['Price']]
        Row_after=Dataset['Rooms'].count()
        Lost=(Row_before-Row_after)
        Percent=int( (Row_before-Row_after)/Row_before*100 )
        Dataset=self.Drop_Coloumn(Dataset,'Price')
        print(f'Removed {Lost} rows which is an {Percent}% dataloss')
        print('Remove NaN Ok')
        return Dataset,Labels
    def One_hot_encoding(self,Dataset,Column_name='CouncilArea'):
        # One hot encoding for selected column
        from sklearn.preprocessing import OneHotEncoder
        import numpy as np
        Data=Dataset[Column_name] ;
        Data=Data.to_numpy()

        Data=Data.reshape(-1, 1)

        Encoder=OneHotEncoder()
        One_hot_data=Encoder.fit_transform(Data) 

        Insert=One_hot_data.toarray()

        Categories=Encoder.categories_[0]
        Categories=Categories.tolist()
        for i in range(np.size(Categories) ): # Add categories
            Single_row=Insert[:,i]
            Dataset[Categories[i]] = Single_row
        print(f'Added {np.size(Categories)} features for {Column_name} -  One-hot Ok')
        return Dataset
    
    def Drop_Coloumn(self,Dataset,Coloumn_name):
        # Drop specified column
        New=Dataset.drop(Coloumn_name, axis='columns', inplace=False);
        return New
    def Convert_data_type(self,Dataset):
        # Convert date to days after Jan 1990 or somoething like that
        from datetime import datetime
        from dateutil.parser import parse
        data=Dataset['Date']
        Array=[]
        for date in data:
            datasime=parse(date)
            int_date=datasime.toordinal()
            Array.append(int_date)
        Dataset.Date=Array
        print('Date set to int type OK')
        return Dataset
    def Preprocess(self,Dataset,Labels,Onehot,Drop,K=3,Plot=False,Edist=False):
        print('Dataset')
        # Main preprocess sript
        # Dataset = X_dateset which you want to transform
        # Labels Corrosponding y to X which also must be transformed because of drop NaN
        # Onehot List of Column names where Onehot encoding is used
        # Drop List of column names which are to be dropped
        # K K for Knearest neighbor 
        # Plot. If true plots Lat/Long with CouncilArea before and after Knearest
        print(type(Dataset))
        if Plot==True:
            self.Plot_Long_lat(Dataset,'CouncilArea')
        # K neigbor to get missing Areacodes
        Dataset=self.K_neighbor_Areacodes(Dataset,K)

        if Plot==True:
            self.Plot_Long_lat(Dataset,'CouncilArea')
        
        # Drop unwanted categories
        for Category in Drop:
            Dataset=self.Drop_Coloumn(Dataset, Category)

        if Edist==True:
            import numpy as np
            import pandas as pd
            print('Ok')            
            Lat=Dataset.Lattitude
            Long=Dataset.Longtitude
            R=np.sqrt((Lat-np.mean(Lat) ) **2 + ( Long - np.mean(Long) ) **2 )
            R=R.to_numpy()
            d = {'E_dist': R}
            df = pd.DataFrame(data=d)
            Dataset=pd.concat([Dataset,df],axis=1)
            Dataset=self.Drop_Coloumn(Dataset, 'Lattitude')

            Dataset=self.Drop_Coloumn(Dataset, 'Longtitude')
        # Drop NaN Rows
        Dataset,Labels=self.Drop_NaN_rows(Dataset,Labels,Row_Col='index') # Can be set to drop Coloumwise if wanted
        
        # Date to number
        Dataset=self.Convert_data_type(Dataset)
        
        # One hot encoding
        for Category in Onehot:
            Dataset=self.One_hot_encoding(Dataset,Category)
            Dataset=self.Drop_Coloumn(Dataset,Category)

        return Dataset,Labels
    def Dataframe_to_np(self,Dataframe):
        # Function to go from dataframe to Np array
        Header=Dataframe.columns
        Np_array=Dataframe.to_numpy()
        return Np_array,Header
    def np_to_Dataframe(self,Np_array,Headers):
        # Function to go from np array to dataframe
        import pandas as pd
        Dataframe=pd.DataFrame(Np_array,columns=Headers)
        return Dataframe


