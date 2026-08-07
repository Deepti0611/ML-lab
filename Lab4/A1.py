import pandas as pd
def load():
    # Load the dataset
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign")
    return df

def datatypes(df):
    return df.info(), df.dtypes
def identify(df):
    print(df.columns)
    print(df["ID"].unique())
    print(df["Year_Birth"].unique())
    print(df.describe())
df=load()
print(df)
info,type=datatypes(df)
print("info")
print(info)
print("type")
print(type)
identify(df)