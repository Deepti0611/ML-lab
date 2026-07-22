import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
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

'''''
def check_categorical(df):
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns

    for column in categorical_columns:
        print(column, ":", df[column].unique())'''

'''''
Based on my thinking- i have found that even though few columns such as-acceptcmp,complain,response etc
when checked for numericla or categorical, the function returns numeric , but in reality - they are categorical data
having binary values(0,1), so they dont require encoding
 dtcustomer(date)can be categoricalor numeric so - they dont require encoding 
 do only marital status and education needs to be encoded


'''

# label encoding for education as input feature (ordinal data)
def encode(df): # encode categorical attributes
    le=LabelEncoder()
    df['Education']=le.fit_transform(df['Education'])
    print(le.classes_)
    return df

# one-hot encoding for marital status as input feature is (nominal data)
def one_hot(df):
    encoder = OneHotEncoder(sparse_output=False)

    encoded_data = encoder.fit_transform(df[['Marital_Status']])
    encoded_df = pd.DataFrame(encoded_data,columns=encoder.get_feature_names_out(['Marital_Status']))
    return encoded_df





df=load_data()


#print(df)
print(" the data types in this dataset are:")
print(check_datatype(df))
print("categorical data types:") 
print(category(df))
print("numeric data types:")
print(numeric(df))
print("After label encoding:")
print("\nOne-Hot Encoded Data:")
print(one_hot(df))
print("\nlabel encoded data:")
print(encode(df))
