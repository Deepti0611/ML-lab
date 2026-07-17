import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data():
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI") #derive data from excel sheet 
    return df
def encode(df):
    encode_df=df.copy()
    for col in encode_df.columns:
        if not pd.api.types.is_numeric_dtype(encode_df[col]):
            encoder=LabelEncoder()
            encode_df[col] = encoder.fit_transform(encode_df[col].astype(str))
    return encode_df

def vector(encode_df):
    vector1 = encode_df.iloc[0].values
    vector2 = encode_df.iloc[1].values
    return vector1,vector2

def cosine_similarity(vector1,vector2):
    dot_p=np.dot(vector1,vector2)
    mag1=np.linalg.norm(vector1)
    mag2=np.linalg.norm(vector2)
    cosine=dot_p/(mag1*mag2)
    return cosine

df = load_data()

encoded_df = encode(df)

vector1, vector2 = vector(encoded_df)

print("Vector 1:", vector1)
print("Vector 2:", vector2)

cosine = cosine_similarity(vector1, vector2)

print("Cosine Similarity =", cosine)

