import numpy as np
import pandas as pd

df=pd.read_csv(r'C:\Users\anubh\OneDrive\Desktop\test\insurance - insurance.csv')

from sklearn.model_selection import train_test_split

x=df.drop(columns=['charges'])
y=df['charges']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=22)

print(x_train.shape)
print(x_test.shape) 

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct=ColumnTransformer([
    ('ohe',OneHotEncoder(sparse_output=False,drop='first'),['sex','smoker','region'])
],remainder='passthrough')

x_train_transformed=ct.fit_transform(x_train)
x_test_transformed=ct.transform(x_test)

print(x_train.shape)
print(x_train_transformed.shape)