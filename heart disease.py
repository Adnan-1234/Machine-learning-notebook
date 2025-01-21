#####import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from sklearn.metrics import classification_report
######read the data
df=pd.read_csv("C:/Users/QHS 006/Downloads/archive (3)/Dataset Heart Disease.csv")
# print(df.head())
# print(df.isnull().sum())
# print(df.columns)
####encode the categorical data
le=LabelEncoder()
for col in df.columns:
    if df[col].dtype =="object" or df[col].dtype=='category':
        df[col]=le.fit_transform(df[col])
print(df.head())

####now split the data into x and y
x=df.drop('target',axis=1)
y=df['target']

####now scale the data to standard value
sc=StandardScaler()
scaledx=sc.fit_transform(x)
###now train the model
xtrain,xtest,ytrain,ytest=train_test_split(scaledx,y,test_size=0.2,random_state=42)
###now make a model
model=RandomForestClassifier(criterion='entropy',max_depth=50,max_features='sqrt',random_state=42)
model.fit(xtrain,ytrain)
pred=model.predict(xtest)
#########now evaluate the model
accuracy=classification_report(ytest,pred)
print(accuracy)

