import pandas as pd
import numpy as np

def create_matrix():
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data")
    '''
    print(df)

    import pandas as pd

    file = pd.ExcelFile("Lab Session Data.xlsx")

    print(file.sheet_names)'''


    X=df[['Candies (#)','Mangoes (Kg)','Milk Packets (#)']].to_numpy()
    y=df[['Payment (Rs)']].to_numpy()
    return X,y
def rank(X):
    from numpy. linalg import matrix_rank 
    rank=matrix_rank(X)
    return rank
def cost_of_product(X):
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