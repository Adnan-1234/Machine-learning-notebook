 #svm se hm apni cheezo ko 1 line ki base pr separate krte ha
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.svm import SVC
df=sns.load_dataset("iris")
x=df.drop("species",axis=1)
y=df["species"]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
model=SVC() ##agar mere output me kuch error a rha ha to me yha ()me kernel = wo rakhoo ga jis pr accuracu 1 ae
model.fit(xtrain,ytrain)
pred=model.predict(xtest)
print(classification_report(ytest,pred))
print(confusion_matrix(ytest,pred))
plt.figure(figsize=(5,5))
sns.heatmap(confusion_matrix(ytest,pred),annot=True)
plt.show()
