#outliers hamare paas data me mistakes hoti ha jo hamare data se bht zyada differ hoti ha mtlb k data me jo values ha height ki wo 10 se 20 k beec ha pr unme koi 1 aisi value bhi ha jo 500 ha ye outlier ha.

# handling include 2 methods z_score and IQR
#z_score:
        #z=x-u/sigma where; x= data point,u=mean of data and sigma=std
#using scipy

import numpy as np
from scipy import stats
data=[10,20,30,330000,40,50,60,70,80,90]
z_score=np.abs(stats.zscore(data))
threshold=2.5
outliers=np.where(z_score<threshold)       # ye hme sara data show kre ga
print(data)

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
data=[10,20,30,330000,40,50,60,70,80,90]
z_score=np.abs(stats.zscore(data))
threshold=2.5
no_outliers=np.where(z_score<threshold)
for i in no_outliers[0]:                  #is se hm apne data me se outliers ko remove kr k data ko print krwae ge
    print(data[i])



#2 method IQR method

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
data=[20,30,40,5000,50,60,70,80,90000]

q1=np.percentile(data,25)
q3=np.percentile(data,75)
iqr=q3-q1
lb=q1-1.5*iqr
up=q3+1.5*iqr
outliers=(np.array(data)>up)|(np.array(data)<lb)
for i in range(len(data)):
    if outliers[i]:                                     #ye hme outlie rka data deta ha

        print(data[i])


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
data=[20,30,40,5000,50,60,70,80,90000]

q1=np.percentile(data,25)
q3=np.percentile(data,75)
iqr=q3-q1
lb=q1-1.5*iqr
up=q3+1.5*iqr
outliers=(np.array(data)>up)|(np.array(data)<lb)
for i in range(len(data)):
    if not outliers[i]:                             #ye hme outlier k begair data deta ha
        print(data[i])
