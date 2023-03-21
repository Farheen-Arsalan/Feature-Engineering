# Missing Value Ratio
import pandas as pd
import numpy as np 

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



print(data.head())
print(pd.get_dummies(data)) # converting the catagorial
#data columns into numbers
