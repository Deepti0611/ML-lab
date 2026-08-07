import pandas as pd
import numpy as np
import math
A=[2,-1,6,4,8]
B=[3,7,2,9,0]
def dot_product(v1, v2):
    dot = 0

    for i in range(len(v1)):
        dot += v1[i] * v2[i]

    return dot
def vector_length(v):
    total = 0

    for value in v:
        total += value ** 2

    return total ** 0.5
def dot_inbuilt(A,B):
    return np.dot(A,B)
def manual_norm(A,B):
    length1 = np.linalg.norm(A)
    length2 = np.linalg.norm(B)
    return length1, length2
dot = dot_product(A, B)
print("Dot Product =", dot)
length1, length2 = manual_norm(A, B)

print("Length of Vector 1 =", length1)
print("Length of Vector 2 =", length2)
print("Dot Product (Inbuilt) =", dot_inbuilt(A, B))
print("Length of Vector 1 (Inbuilt) =", np.linalg.norm(A))
print("Length of Vector 2 (Inbuilt) =", np.linalg.norm(B))