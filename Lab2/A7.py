import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
def load_data(): # load data 
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI",na_values=["?"]) #derive data from excel sheet 
    return df

def binary(df):# function to identify binary categories
    first_twenty=df.iloc[:20]
    binary=[]
    for col in df.columns:
        unique_val=df[col].dropna().unique() # identify columns with binary attributes
        if len(unique_val)==2:
            binary.append(col)
    binary_data=first_twenty[binary].copy()
    for col in binary:
        unique_val=df[col].dropna().unique()
        if not set(unique_val).issubset({0,1}):
            mapping={
                unique_val[0]:0,unique_val[1]:1 # convert binary values to 0 and 1 if necessary
            }
            binary_data[col]=binary_data[col].map(mapping)
    return binary_data
def encode(df): # encode categorical attributes
    encode_df=df.iloc[:20].copy()
    encode_df=encode_df.drop(columns=["Record ID","Condition"]) # dropping these columns as they dont contribute largely in cosine similarity
    for col in encode_df.columns:
        if encode_df[col].dtype == 'object' or encode_df[col].dtype == 'str':
            encoder=LabelEncoder()
            encode_df[col] = encoder.fit_transform(encode_df[col].astype(str))
    encode_df = encode_df.fillna(0)
    return encode_df



def calculate(vector1,vector2): # function to compute f00,f01,f10,f11
    f00 = 0
    f01 = 0
    f10 = 0
    f11 = 0

    for i in range(len(vector1)):
        if vector1[i] == 0 and vector2[i] == 0:
            f00 += 1
        elif vector1[i] == 0 and vector2[i] == 1:
            f01 += 1
        elif vector1[i] == 1 and vector2[i] == 0:
            f10 += 1
        elif vector1[i] == 1 and vector2[i] == 1:
            f11 += 1
    return f00,f01,f10,f11

def Jaccard(f00,f01,f10,f11): # function to caluclate  JC and SMC
    denominator_jc=f01+f10+f11
    denominator_smc=f00+f01+f10+f11
    if denominator_jc==0: # to avoid division by 0
        JC=1
    else:
        JC=f11/denominator_jc
    if denominator_smc==0:
        SMC=1
    else:
        SMC=(f11+f00)/denominator_smc
    return JC,SMC
def cosine_similarity(vector1,vector2): # cosine similarity calculation
    dot_p=np.dot(vector1,vector2)
    mag1=np.linalg.norm(vector1)
    mag2=np.linalg.norm(vector2)
    cosine=dot_p/(mag1*mag2)
    return cosine

def similarity_matrices(binary_data,encode_df): # similarity matrix generation using JC,SMC and cosine similarity
    JC_matrix = np.zeros((20, 20))
    SMC_matrix = np.zeros((20, 20))
    COS_matrix = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
             binary_vector1 = binary_data.iloc[i].values
             binary_vector2 = binary_data.iloc[j].values
             f00, f01, f10, f11 = calculate(binary_vector1, binary_vector2)
             JC, SMC = Jaccard(f00, f01, f10, f11)
             JC_matrix[i][j] = JC
             SMC_matrix[i][j] = SMC
             vector1 = encode_df.iloc[i].values
             vector2 = encode_df.iloc[j].values
             COS_matrix[i][j]=cosine_similarity(vector1,vector2)
    return JC_matrix, SMC_matrix, COS_matrix
def heatmap(JC_matrix,SMC_matrix,COS_matrix): # function to generate heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(JC_matrix, annot=True,fmt=".2f")
    plt.title("Jaccard Coefficient Heatmap")
    plt.show()
    plt.figure(figsize=(10, 8))
    sns.heatmap(SMC_matrix, annot=True,fmt=".2f")
    plt.title("Simple Matching Coefficient Heatmap")
    plt.show()
    plt.figure(figsize=(10, 8))
    sns.heatmap(COS_matrix, annot=True,fmt=".2f")
    plt.title("Cosine Similarity Heatmap")
    plt.show()
df = load_data()

binary_df = binary(df)

encoded_df = encode(df)

JC_matrix, SMC_matrix, COS_matrix = similarity_matrices(binary_df,encoded_df)
heatmap(JC_matrix,SMC_matrix,COS_matrix)