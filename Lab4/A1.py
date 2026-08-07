from json import encoder

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def load():
    # Load the dataset
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign")
    return df

def datatypes(df):
    return df.info(), df.dtypes
def identify(df):
    print(df.columns)
    print(df["Education"].unique())
    print(df["Year_Birth"].unique())
    print(df.describe())
def Onehot(df):
# One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=["Marital_Status"], drop_first=False)

# Display only the encoded columns
    encoded_columns = [col for col in df_encoded.columns
                   if col.startswith("Marital_Status_")]

    return (df_encoded[encoded_columns])
def label_encode(df):
    education_order = {
    "Basic": 0,
    "Graduation": 1,
    "2n Cycle":2,
    "Master": 3,
    "PhD": 4
}

    df["Education_Encoded"] = df["Education"].map(education_order)

    mapping_df = pd.DataFrame(
        list(education_order.items()),
        columns=["Original Value", "Encoded Value"]
)

    return (mapping_df)

df=load()
print(df)
info,type=datatypes(df)
print("info")
print(info)
print("type")
print(type)
identify(df)
print()
print("one hot encoded-marital status")
print(Onehot(df))
print()
print("label encoded-education")
print(label_encode(df))