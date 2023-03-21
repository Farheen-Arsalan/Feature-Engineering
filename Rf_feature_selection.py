# random forest as feature selector 
import pandas as pd
import numpy as np 
from sklearn import ensemble 
from sklearn import preprocessing 

path = '/Users/Arsalan/Desktop/train_v9rqX0R.csv'
data = pd.read_csv(path)
print(data.shape)
print(data.columns)
# find how much percentage of the data is null 


# number of missing values 
b = data.isnull().sum()
print(b)

bper = ((data.isnull().sum())/data.shape[0])*100
print(bper)

# fill the missing values 
# mean 
mn = data['Item_Weight'].mean() # number columns
mode = data['Outlet_Size'].mode()[0] # categorical columns
print(mode)

data['Item_Weight'].fillna(mn,inplace=True)
data['Outlet_Size'].fillna(mode,inplace=True)
bper = ((data.isnull().sum())/data.shape[0])*100
print(bper)

# find the variance in the data 
# vairance is only for the numerical values
# not for catagorical
#data1 = pd.get_dummies(data)


#nX = pd.get_dummies(X)

leModel = preprocessing.LabelEncoder()
data['Item_Identifier']= leModel.fit_transform(data['Item_Identifier'])
leModel = preprocessing.LabelEncoder()
data['Item_Fat_Content']= leModel.fit_transform(data['Item_Fat_Content'])
leModel = preprocessing.LabelEncoder()
data['Item_Type']= leModel.fit_transform(data['Item_Type'])
leModel = preprocessing.LabelEncoder()
data['Outlet_Identifier']= leModel.fit_transform(data['Outlet_Identifier'])
leModel = preprocessing.LabelEncoder()
data['Outlet_Size']= leModel.fit_transform(data['Outlet_Size'])
leModel = preprocessing.LabelEncoder()
data['Outlet_Location_Type']= leModel.fit_transform(data['Outlet_Location_Type'])
leModel = preprocessing.LabelEncoder()
data['Outlet_Type']= leModel.fit_transform(data['Outlet_Type'])

Y = data['Item_Outlet_Sales'] 
X = data.drop('Item_Outlet_Sales',axis=1)

# create the regresson RF model 
rfModel = ensemble.RandomForestRegressor(max_depth=5,random_state=2)
rfModel = rfModel.fit(X,Y)


ind = np.argsort(rfModel.feature_importances_)
print(ind)

indImportant = ind[:7] # top 10 indexs
features = X.columns
importan_features = features[indImportant]
print(importan_features)









