import pandas as pd
import numpy as np 
from sklearn import linear_model
from sklearn.feature_selection import f_regression # forward feature selection
from sklearn import preprocessing
from sklearn import decomposition

path ='/Users/Arsalan/Desktop/train_v9rqX0R.csv'
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

leModel = preprocessing.LabelEncoder()
X['Item_Fat_Content']= leModel.fit_transform(X['Item_Fat_Content'])

X['Item_Type']= leModel.fit_transform(X['Item_Type'])


X['Outlet_Size']= leModel.fit_transform(X['Outlet_Size'])

X['Outlet_Location_Type']= leModel.fit_transform(X['Outlet_Location_Type'])

X['Outlet_Type']= leModel.fit_transform(X['Outlet_Type'])

print("Len of the dimension: ",len(X.columns))
# reduce the dimensions

pcaModel = decomposition.PCA(n_components=4,random_state=1)

Xupdated = pcaModel.fit_transform(X)

# ICA 
ICAModel = decomposition.FastICA(n_components=4,random_state=1)
Xupdated1 = pcaModel.fit_transform(X)
print(Xupdated1.shape)


print(Xupdated.shape)




