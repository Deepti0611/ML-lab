from cv2 import kmeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
def load_data(): # load data 
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign",na_values=["?"]) #derive data from excel sheet 
    return df

def choose_columns(df):
    # Select two features
    X = df[["Income", "MntWines"]].dropna()

# Convert to matrix
    points = X.to_numpy()

    return points

def k_means(points):
    centroid1 = points[0]
    centroid2 = points[20]
    centroid3 = points[50]
    while True:
        c1 = []
        c2 = []
        c3 = []
        for point in points:
            d1 = np.linalg.norm(point - centroid1)
            d2 = np.linalg.norm(point - centroid2)
            d3 = np.linalg.norm(point - centroid3)
            minimum = min(d1, d2, d3)
            if minimum == d1:
                c1.append(point)
            elif minimum == d2:
                c2.append(point)
            else:
                c3.append(point)
        new_centroid1 = np.mean(c1, axis=0)
        new_centroid2 = np.mean(c2, axis=0)
        new_centroid3 = np.mean(c3, axis=0)
        # Check if centroids changed
        if np.allclose(new_centroid1, centroid1) and np.allclose(new_centroid2, centroid2) and np.allclose(new_centroid3, centroid3):
            break
        centroid1 = new_centroid1
        centroid2 = new_centroid2
        centroid3 = new_centroid3
    return c1, c2, c3, new_centroid1, new_centroid2, new_centroid3

df=load_data()
points=choose_columns(df)
c1, c2, c3, new_centroid1, new_centroid2, new_centroid3 = k_means(points)
print("Cluster 1 points:")
print(c1)
print("Cluster 2 points:")
print(c2)
print("Cluster 3 points:")
print(c3)
print("New Centroid 1:", new_centroid1)
print("New Centroid 2:", new_centroid2)
print("New Centroid 3:", new_centroid3)


    

