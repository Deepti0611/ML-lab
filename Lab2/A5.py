import pandas as pd
def load_data():
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI") #derive data from excel sheet 
    return df

def binary(df):
    first_two=df.iloc[:2]
    binary=[]
    for col in df.columns:
        unique_val=df[col].dropna().unique()
        if len(unique_val)==2:
            binary.append(col)
    binary_data=first_two[binary].copy()
    for col in binary:
        unique_val=df[col].dropna().unique()
        if not set(unique_val).issubset({0,1}):
            mapping={
                unique_val[0]:0,unique_val[1]:1
            }
            binary_data[col]=binary_data[col].map(mapping)
    return binary_data

def vector(binary_data):
    vector1=binary_data.iloc[0].values
    vector2=binary_data.iloc[1].values
    return vector1,vector2

def calculate(vector1,vector2):
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

def Jaccard(f00,f01,f10,f11):
    JC = f11 / (f01 + f10 + f11)
    SMC = (f11 + f00) / (f00 + f01 + f10 + f11)
    return JC,SMC

df=load_data()
binary_data=binary(df)
vector1,vector2=vector(binary_data)
print("Vector 1:", vector1)
print("Vector 2:", vector2)
f00,f01,f10,f11=calculate(vector1,vector2)
print("f00 =", f00)
print("f01 =", f01)
print("f10 =", f10)
print("f11 =", f11)

JC,SMC=Jaccard(f00,f01,f10,f11)

print("Jaccard Coefficient (JC) =", JC)
print("Simple Matching Coefficient (SMC) =", SMC)

if JC > SMC:
    print("JC is greater than SMC")
elif SMC > JC:
    print("SMC is greater than JC")
else:
    print("JC and SMC are equal")





