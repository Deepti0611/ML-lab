import pandas as pd
import numpy as np
from collections import Counter
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
            data=impute_df[col].to_numpy()
            filtered_data=[ x for x in data if pd.notna(x)]
            mode_val=Counter(filtered_data).most_common(1)[0][0]
            impute_df[col]=[x if pd.notna(x) else mode_val for x in data]
    return impute_df

df=load_data()

print("Missing values before imputation:")
print(df.isnull().sum())

imputed_df = impute(df)

print("\nMissing values after imputation:")
print(imputed_df.isnull().sum())



    
