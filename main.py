import numpy as np
import pandas as pd

df=pd.read_csv(r'C:\Users\anubh\OneDrive\Desktop\test\insurance - insurance.csv')

from sklearn.model_selection import train_test_split

x=df.drop(columns=['charges'])
y=df['charges']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=22)

print(x_train.shape)
print(x_test.shape)