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

thresholdMissingValue = 10 # 10 Percentage
# now we can fix what percentage of missing values should be discarded
# as the legitimate column
columns = data.columns
selectedColumns =[]

ind   = bper<=thresholdMissingValue
print(ind)

selectedColumns = columns[ind]
print(selectedColumns)
