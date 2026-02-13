import numpy as np
import pandas as pd

df=pd.read_csv(r'C:\Users\anubh\OneDrive\Desktop\test\insurance - insurance.csv')

from sklearn.model_selection import train_test_split

x=df.drop(columns=['charges'])
y=df['charges']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=22)

from sklearn.preprocessing import OrdinalEncoder
oe=OrdinalEncoder(categories=[['yes','no']])
x_train_smoker=oe.fit_transforrm(x_train[['smoker']])
x_test_smoker=oe.transform(x_test[['smoker']])

ohe=OneHotEncoder(drop='first',sparse_output=False)
x_train_gender_region=ohe.fit_transform(x_train[['sex','region']])
x_test_gender_region=ohe.transform(x_test[['sex','region']])

x_train_num=x_train.drop(columns=['sex','smoker','region']).values
x_test_num=x_test.drop(columns=['sex','smoker','region']).values

x_train_f=np.concatenate([x_train_num,x_train_gender_region,x_train_smoker],axis=1)
x_test_f=np.concatenate([x_test_num,x_test_gender_region,x_test_smoker],axis=1)

print(x_train_f.shape)
print(x_test_f.shape) 

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct=ColumnTransformer([
    ('ohe',OneHotEncoder(sparse_output=False,drop='first'),['sex','region']),
    ('ord',OrdinalEncoder(categories=[['yes','no']]),['smoker'])
],remainder='passthrough')

x_train_transformed=ct.fit_transform(x_train)
x_test_transformed=ct.transform(x_test)

print(x_train_transformed.shape)
print(x_test_transformed.shape)