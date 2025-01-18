#######import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import tensorflow as tf


###load the data
df=pd.read_csv('C:\\Users\\QHS 006\\Downloads\\archive (2)\\global_cancer_predictions.csv')
print(df.isnull().sum().sort_values(ascending=True))
##print(df.head())


##now encode the data
le=LabelEncoder()
for col in df.columns:
    if df[col].dtype=='object' or df[col].dtype=='category':
        df[col]=le.fit_transform(df[col])


##split the data in x and y
x=df.drop('Cancer_Type',axis=1)
y=df['Cancer_Type']
##now scale the data
scale=StandardScaler()
scaled_df=scale.fit_transform(x)
##now train and test the model
xtrain,xtest,ytrain,ytest=train_test_split(scaled_df,y,test_size=0.2,random_state=42)
##create a neyral network
il=tf.keras.layers.Dense(100,activation='relu',input_shape=(xtrain.shape[1],))
hl=tf.keras.layers.Dense(50,activation='relu')
ol=tf.keras.layers.Dense(10,activation='softmax')
model=tf.keras.models.Sequential([il,hl,ol])
###now compile the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(xtrain,ytrain,epochs=10,batch_size=32,verbose=1)
##now predict the value
pred_prob=model.predict(xtest)
pred = np.argmax(pred_prob, axis=1)

###now evaluate the model
accuracy=classification_report(ytest,pred)
print(accuracy)


