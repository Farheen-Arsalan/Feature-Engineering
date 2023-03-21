import pandas as pd
import numpy as np 
from sklearn import linear_model
from sklearn.feature_selection import RFE # reccursive feature eliminaiton
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

X  = data.drop(['Item_Outlet_Sales'],axis=1)
X = X.drop(['Outlet_Identifier','Item_Identifier'],axis=1)
Y = data['Item_Outlet_Sales']

print(X.shape)
print(Y.shape)

leModel = preprocessing.LabelEncoder()
X['Item_Fat_Content']= leModel.fit_transform(X['Item_Fat_Content'])

X['Item_Type']= leModel.fit_transform(X['Item_Type'])


X['Outlet_Size']= leModel.fit_transform(X['Outlet_Size'])
X['Outlet_Location_Type']= leModel.fit_transform(X['Outlet_Location_Type'])

X['Outlet_Type']= leModel.fit_transform(X['Outlet_Type'])
print(X.head())

# create the model 
 
modelLm = linear_model.LinearRegression()
rfeModel = RFE(modelLm,n_features_to_select=4)
rfeModel = rfeModel.fit(X,Y)

print(rfeModel.support_)
print(rfeModel.ranking_)

features = X.columns 
SelectedFEatures = features[rfeModel.support_]
print("All the features: ")
print(features)

print("Selected features: ")
print(SelectedFEatures)
