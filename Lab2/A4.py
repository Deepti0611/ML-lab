import pandas as pd
import numpy as np
def load_data(): # load data 
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI",na_values=["?"]) #derive data from excel sheet 
    return df

def check_datatype(df): #function to identify datatype
    return df.dtypes
def encoding_cat(df): # function to identify encoding method
    categorical_columns = df.select_dtypes(include=['object', 'str']).columns
    encoding={}
    for col in categorical_columns:
        unique_val=df[col].dropna().unique() # binary encoding and one-hot encoding categorization
        if len(unique_val)==2:
             encoding[col] = "Binary Encoding"
        else:
            encoding[col] = "One-Hot Encoding"
    return encoding

def find_range(df): # idenitfy range of numeric attributes
    numeric_cols=df.select_dtypes(include=np.number).columns # extract numeric attribute column
    ranges={}
    for col in numeric_cols:
        min=df[col].min()
        max=df[col].max()
        ranges[col] = (min, max)
    return ranges
def find_missing(df): # function to find missing values
    return df.isnull().sum()
def find_outliers(df): # function to find outliers 
    numeric_cols = df.select_dtypes(include=np.number).columns
    outlier_count={} # to store detected outlier count
    for col in numeric_cols:
        Q1=df[col].quantile(0.25)
        Q3=df[col].quantile(0.75)
        IQR=Q3-Q1 # Iqr method to find outliers
        lb=Q1-1.5*IQR
        ub=Q3+1.5*IQR
        outliers=df[((df[col]<lb) |(df[col]>ub))]
        outlier_count[col] = len(outliers)
    return outlier_count

def mean_varience(df): # function to find mean and variance
      numeric_cols = df.select_dtypes(include=np.number).columns
      result={}
      for col in numeric_cols:
          mean=df[col].mean()
          variance=df[col].var()
          result[col]=(mean,variance)
      return result

df = load_data()

print("Data types are:")
print(check_datatype(df))

print("Encoding schemes are:")
print(encoding_cat(df))

print("Ranges are:")
print(find_range(df))

print("Missing values are:")
print(find_missing(df))

print("Number of outliers are:")
print(find_outliers(df))

print("Mean and Variance are:")
print(mean_varience(df))

    
          



    
    



