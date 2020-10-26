# -*- coding: utf-8 -*-
"""Soft_Computing_IA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZRzvmexx4k4o0yjsJMGX0Ngh5ISUvCvR

# Customer Segmentation using Kohonen Self Organising Maps

### Importing important packages
"""
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


"""### Update winning vector"""
def update(weights,dftuple,w,alpha):
    x=dftuple-weights[w]
    x=x*alpha
    weights[w]=x+weights[w]
    return weights



"""### Random initial weights"""
weights=[np.random.randn(df1.shape[1]),np.random.randn(df1.shape[1])]
weights



"""### Driver Code"""
epochs=100
alpha=0.15
final=np.zeros(dataset.shape[0])
for i in range(epochs):
    for j in range(dataset.shape[0]):
        w=compute(weights,dataset[j])
        if i==epochs-1:
            final[j]=w
        weights=update(weights,dataset[j],w,alpha)
    alpha=0.5*alpha*(i/(i+1))
final



"""### Adding the assigned cluster as a column to the dataframe"""
df['cluster']=final
df.head()

df.groupby('cluster').mean()

area=np.pi*(X[:,0])**1.2 
plt.scatter(X[:,0],X[:, 3],s=area,c=final.astype(np.float),alpha=0.5)
plt.xlabel('Age',fontsize=18)
plt.ylabel('Income',fontsize=16)
plt.show()

fig=plt.figure(1,figsize=(8,6))
plt.clf()
ax=Axes3D(fig,rect=[0,0,.95,1],elev=48,azim=134)
plt.cla()
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')
ax.scatter(X[:,1],X[:,0],X[:,3],c=final.astype(np.float))

