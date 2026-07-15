import time

import pandas as pd
import numpy as np

def load_data(): # load data 
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price") #derive data from excel sheet 
    return df
def mean_varience_numpy(df): # function to find mean and varience of stock price 
    mean=np.mean(df['Price'])
    varience=np.var(df['Price'])
    return mean,varience
def mean_varience(df): # function to find mean and varience of stock price manually
    X=df[['Price']].to_numpy()
    n=len(X)
    mean=sum(X)/n #mean of stock price
    varience=sum((X-mean)**2)/n # varience
    return mean,varience
def exec_time(function,df):
    total_time=0
    for i in range(10):# 10 runs 
        start=time.time()
        function(df)
        end=time.time()
        total_time+=end-start# total exexecution time for 10 runs
    avg_time=total_time/10 # average execution time for 10 runs
    return total_time, avg_time
   
print("IRCTC Stock Price Data:")
df = load_data()
print(df)
mean,varience=mean_varience_numpy(df)
print("Mean of Stock Price:",mean)
print("Varience of Stock Price:",varience)
mean1,varience1=mean_varience(df)
print("Mean of Stock Price:",mean1)
print("Varience of Stock Price:",varience1)
total_time_numpy, avg_time_numpy=exec_time(mean_varience_numpy,df)
total_time_manual, avg_time_manual=exec_time(mean_varience,df)
print("numpy execution time:",total_time_numpy)
print("manual execution time:",total_time_manual)
print("mean difference:",abs(mean-mean1))
print("varience difference:",abs(varience-varience1))
print("average execution time for numpy:",avg_time_numpy)
print("average execution time for manual:",avg_time_manual)