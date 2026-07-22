import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
def load_data(): # load data 
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign",na_values=["?"]) #derive data from excel sheet 
    return df

def check_datatype(df): #function to identify datatype
    return df.dtypes

def category(df): # function to identify categorical data columns
     categorical_columns = df.select_dtypes(include=['object', 'str']).columns
     return categorical_columns
def numeric(df): # function to identify numerical data columns 
    numeric_columns=df.select_dtypes(include=np.number).columns
    return numeric_columns

def encode(df): # encode categorical attributes
    encode_df=df.copy()
    for col in encode_df.columns:
        if not pd.api.types.is_numeric_dtype(encode_df[col]):
            encoder=LabelEncoder()
            encode_df[col] = encoder.fit_transform(encode_df[col].astype(str))
    return encode_df
'''''
def check_categorical(df):
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns

    for column in categorical_columns:
        print(column, ":", df[column].unique())'''

df=load_data()

#print(df)
print(" the data types in this dataset are:")
print(check_datatype(df))
print("categorical data types:") 
print(category(df))
print("numeric data types:")
print(numeric(df))
print(encode)
