import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def load_data(): # load data 
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign",na_values=["?"]) #derive data from excel sheet 
    return df
def choose_column(df):
    feature=df["Income"].dropna()
    return feature
def mean_variance(feature):
    mean=np.mean(feature)
    var=np.var(feature)
    return mean,var
def histogram(feature):
    hist, bins = np.histogram(feature, bins =10)
    print(hist)
    print(bins)
    plt.hist(feature,bins=10)
    plt.xlabel("Income")
    plt.ylabel("Frequency")
    plt.show()

df=load_data()
feature=choose_column(df)
mean,var=mean_variance(feature)
print("mean of income column is :", mean)
print("variance of income column is :",var)
histogram(feature)

