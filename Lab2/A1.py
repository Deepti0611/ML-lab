import pandas as pd
import numpy as np

def create_matrix(): # function to create  matrix 
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data") #derive data from excel sheet 
    '''
    print(df)

    '''
    X=df[['Candies (#)','Mangoes (Kg)','Milk Packets (#)']].to_numpy() # input feature matrix 
    y=df[['Payment (Rs)']].to_numpy() # payment feature 
    return X,y
def rank(X): # function to find rank of matrix
    from numpy. linalg import matrix_rank 
    rank=matrix_rank(X)
    return rank
def cost_of_product(X): # function to find c in Xc=y
    from numpy.linalg import pinv
    a = pinv(X)
    return np.dot(a,y)
    

X, y = create_matrix()
print("Matrix X:")
print(X)
print("Matrix y:")
print(y)
print("Rank of Matrix X:")
print(rank(X))
print("Cost of Product:")
print(cost_of_product(X))