import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
def load_data():
    df = pd.read_excel("Lab Session Data.xlsx",sheet_name="thyroid0387_UCI",na_values=["?"])
    return df

def impute(df):
    impute_df=df.copy()
    for col in impute_df.columns:
        if pd.api.types.is_any_real_numeric_dtype(impute_df[col]):
            data=impute_df[col].to_numpy(dtype=float)
            q1=np.nanpercentile(data,25)
            q3=np.nanpercentile(data,75)
            IQR=q3-q1
            lower_bound=q1-1.5*IQR
            upper_bound=q3+1.5*IQR
            outliers=data[(data<lower_bound)|(data>upper_bound)]
            if len(outliers)>0:
                value=np.nanmedian(data)
            else:
                value=np.nanmean(data)
            impute_df[col]=np.where(pd.isna(data),value,data)
        else:
            mode_val=impute_df[col].mode()[0]
            impute_df[col]=impute_df[col].fillna(mode_val)
            
    return impute_df
def numeric(df):
    normalized=df.copy()
    numeric_cols=df.select_dtypes(include=np.number).columns.tolist()
    if "Record ID" in numeric_cols:
        numeric_cols.remove("Record ID")
    return numeric_cols

def normalize(df):
    normalized=df.copy()
    numeric_cols=numeric(df)
    scalar=MinMaxScaler()
    normalized[numeric_cols]=scalar.fit_transform(normalized[numeric_cols])
    return normalized


df=load_data()
imputed_df = impute(df)
normalized_df=normalize(imputed_df)

print("Data before normalization:")
print(imputed_df.head())

print("Data after normalization:")
print(normalized_df.head())


