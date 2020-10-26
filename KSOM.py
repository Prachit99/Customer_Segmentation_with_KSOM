import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


"""### Reading the dataset"""
df=pd.read_csv('Cust_Segmentation.csv')
df.head()


"""## Data Preprocessing
### Removing address column as it is not numeric
"""
df=df.drop('Address',axis=1)
df1=df.drop('Customer Id',axis=1)
df1.head()


"""### Normalizing all values in the table
X=df1.values[:,0:]
X=np.nan_to_num(X)
dataset=StandardScaler().fit_transform(X)
dataset


## Clustering using Kohonen SOM
### Computing winning vector using Euclidean Distance
"""
def compute(weights,dftuple):
    x=dftuple-weights[0]
    y=dftuple-weights[1]
    x=np.power(x,2)
    y=np.power(y,2)
    a=np.sum(x)
    b=np.sum(y)
    if a>b:
        return 0
    else:
        return 1
