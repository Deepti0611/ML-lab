import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
def load_data(): # load data 
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign",na_values=["?"]) #derive data from excel sheet 
    return df
def minkowski_manual(a,b,p):
    total=0
    for i in range(len(a)):
        x=abs(a[i]-b[i])
        total=total+pow(x,p)
    return pow(total,1/p)
def eucledean(a,b):
    return minkowski_manual(a,b,2)
def manhattan(a,b):
    return minkowski_manual(a,b,1)

def vectors(df):
    numeric_columns=df.select_dtypes(include=np.number)
    a=numeric_columns.iloc[0].to_numpy()
    b=numeric_columns.iloc[1].to_numpy()
    return a,b
def plot(a,b):
    x=[]
    p=[]
    for i in range(1,len(a)):
        ans=minkowski_manual(a,b,i)
        x.append(ans)
        p.append(i)
    plt.plot(p,x,marker='o')
    plt.xlabel("order of p")
    plt.ylabel("minkowski value")
    plt.show()
    
def inbuiltplot(a,b):
    x=[]
    p=[]
    for i in range(1,len(a)):
        ans=distance.minkowski(a,b,i)
        x.append(ans)
        p.append(i)
    plt.plot(p,x,marker='o')
    plt.xlabel("order of p")
    plt.ylabel("minkowski value")
    plt.show()
df=load_data()
a,b=vectors(df)
print("\n manhattan distance:")
print(manhattan(a,b))
print("\neucledean distance")
print(eucledean(a,b))
plot(a,b)
inbuiltplot(a,b)
