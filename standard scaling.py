# scaling ka main purpose ye hota ha k hm apne data ko 1 mean value pr le ae ta k ml model k liye asani ho data ko read krne k liye
#agar hm scaling na kre to phir jo zyada value hogi usko prediction k liye zyada favour mile gi or prediction wrong ae gi.

#standard scaling.iski value -3 se +3 k b/w ae gi.
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
data={
    "weight":[50,70,65,80,170,200],
    "height":[120,140,135,180,134,127],
    "age":[20,25,24,30,26,28]
}
df=pd.DataFrame(data)
scaler=StandardScaler()
fit=scaler.fit_transform(df)
print(fit)

# min-max scaler.iski value 0 se 1 k b/w hogi.ye specific range k liye use kre jb hme 0 and 1 k b/w chahiye value.
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
data={
    "weight":[50,70,65,80,170,200],
    "height":[120,140,135,180,134,127],
    "age":[20,25,24,30,26,28]
}
df=pd.DataFrame(data)
scaler=MinMaxScaler()
fit=scaler.fit_transform(df)
print(fit)

#abs scaler or robust scaler.agar mere data me outliers bhi ha to isse use kre q k ye outliers ko bhi remove krta ha.iski value ki koi range nhi ha negative bhi ho skti ha.


import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
data={
    "weight":[50,70,65,80,170,200],
    "height":[120,140,135,180,134,127],
    "age":[20,25,24,30,26,28]
}
df=pd.DataFrame(data)
scaler=RobustScaler()
fit=scaler.fit_transform(df)
print(fit)

#max-abs scaler.isse hm tb use krte ha jb hmne spare mtlb OS data k saath kaam krna ho ya data ko mean=0 na lana ho.

import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
data={
    "weight":[50,70,65,80,170,200],
    "height":[120,140,135,180,134,127],
    "age":[20,25,24,30,26,28]
}
df=pd.DataFrame(data)
scaler=MaxAbsScaler()
fit=scaler.fit_transform(df)
print(fit)