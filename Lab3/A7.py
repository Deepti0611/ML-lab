import pandas as pd
import numpy as np
import math
A=[2,-1,6,4,8]
B=[3,7,2,9,0]
def manual_dotproduct(A,B):
    total=0
    for i in range(len(A)):
        total=total+A[i]*B[i]
    return total
def dot_inbuilt(A,B):
    return np.dot(A,B)
def manual_norm(A,B):
    a=len(A)
    b=len(B)
    total1=0
    total2=0
    for i in range(a):
        total1=total1+A[i]**2
        total2=total2+B[i]**2
    return total1**0.5,total2**0.5

def builtinnorm(A,B):
    return np.linalg.norm(A),np.linalg.norm(B)



print(manual_dotproduct(A,B))
print(dot_inbuilt(A,B))
print(manual_norm(A,B))
print(builtinnorm(A,B))

