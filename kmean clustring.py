from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
#generate sample data
x,y=make_blobs(n_features=1000,n_samples=13,centers=3,random_state=42)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)

wcss=[]
for k in range(1,11):
    kmean=KMeans(n_clusters=k,init="k-means++",random_state=42)
    kmean.fit(xtrain)
    wcss.append(kmean.inertia_)
##pip install kneed 
from kneed import KneeLocator
a=KneeLocator(range(1,11),wcss,curve="convex")
print(a.elbow)

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from kneed import KneeLocator
x,y=make_blobs(n_features=1000,n_samples=13,centers=3,random_state=42)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,)
wccs=[]
for k in range(1,11):
    kmean=KMeans(n_clusters=k,init="k-means++",random_state=42)
    kmean.fit(xtrain)
    wccs.append(kmean.inertia_)
a=KneeLocator(range(1,11),wccs,curve="convex")
print(a.elbow)

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from kneed import KneeLocator
import seaborn as sns
import matplotlib.pyplot as plt
x,y=make_blobs(n_features=1000,n_samples=40,centers=5,random_state=42)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
wccs=[]
for k in range(1,31):
    kmean=KMeans(n_clusters=k,init="k-means++",random_state=42)
    kmean.fit(xtrain)
    wccs.append(kmean.inertia_)
a=KneeLocator(range(1,31),wccs,curve="convex")
print(a.elbow)
plt.scatter(range(1,31),wccs)
plt.show()
from sklearn.metrics import silhouette_score
sil=[]
for k in range(2,31):
    kmean=KMeans(n_clusters=k,init="k-means++")
    kmean.fit(xtrain)
    score=silhouette_score(xtrain,kmean.labels_)
    sil.append(score)
print(sil)
plt.plot(range(2,31),sil)
plt.show()

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import seaborn as sns
import matplotlib.pyplot as plt
x,y=make_blobs(n_features=2,n_samples=30,centers=4,random_state=42)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
wccs=[]
for k in range(1,17):
    kmean=KMeans(n_clusters=k,init='k-means++',random_state=42)
    kmean.fit(xtrain)
    wccs.append(kmean.inertia_)
from kneed import KneeLocator
a=KneeLocator(range(1,17),wccs,curve='convex')
# print(a.elbow)
# plt.scatter(range(1,17),wccs)
# plt.show()
##ab jo uper cluster ki value nikali ha use model predict kre ge
from sklearn.metrics import silhouette_score
kmean=KMeans(n_clusters=4,random_state=42)
sil=silhouette_score(x,kmean.fit_predict(x))
print(sil)


#########################             DBSCAN                   #########################
###ye non linear data k liye best ha
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
x,y=make_blobs(n_samples=2000,n_features=2,random_state=42,centers=3)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
scale=StandardScaler()
a=scale.fit_transform(x)
dbscan=DBSCAN(eps=0.9)
dbscan.fit(a)
print(dbscan.labels_)
plt.scatter(x[:,0],x[:,1],c=dbscan.labels_)
plt.show()
from sklearn.metrics import silhouette_score
sil=silhouette_score(x,dbscan.fit_predict(x))
print(sil)