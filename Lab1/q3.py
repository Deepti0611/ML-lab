'''import numpy as np

R = 3
C = 3
# another way to create matrix 
print("Enter values space-separated:")
vals = list(map(int, input().split()))

mat = np.array(vals).reshape(R, C)
print(mat)'''
import numpy
from numpy.linalg import matrix_power
def input_matrix(r,c,mat):# user input for matrix
    for i in range(r):
        for j in range(c):
            mat[i][j]=int(input(f"enter values for matrix position {i+1},{j+1}:"))
    
    
    return mat
def powermat(mat,power):# finding power using numpy 
     final_matrix=matrix_power(mat,power)
     return final_matrix 

r=int(input("enter number of rows and columns:"))
c=r
mat = [[0 for j in range(c)] for i in range(r)]
input_matrix(r,c,mat)
for i in range(r):
        for j in range(c):
            print(mat[i][j],end=" ")
        print()
pow=int(input("enter power number:"))
print("matrix after power is:")
final=powermat(mat,pow)
for i in range(r):
    for j in range(c):
        print(final[i][j],end=" ")
    print()

