# Low Variance Filter 
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

# find the variance in the data 
# vairance is only for the numerical values
# not for catagorical
data1 = pd.get_dummies(data)
variance = data1.var()
print(variance)
print()
print()
print(variance.index)
thresholdVariance = 10 # 10 Percentage
# now we can fix what percentage of missing values should be discarded
# as the legitimate column
columns = variance.index
selectedColumns =[]
print()
print()
ind = variance>thresholdVariance
selectedColumns=columns[ind]
print(selectedColumns)






