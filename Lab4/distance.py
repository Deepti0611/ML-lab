import pandas as pd
import numpy as np
from scipy.spatial.distance import minkowski
def load():
    # Load the dataset
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign")
    return df
def extract_numeric(df):
# Keep only numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Select two vectors (rows)
    vector1 = numeric_df.iloc[0].values
    vector2 = numeric_df.iloc[1].values
    return vector1, vector2
    
def calculate_distances(vector1, vector2, p):

    v1 = np.array(vector1)
    v2 = np.array(vector2)

    distance = np.sum(np.abs(v1 - v2) ** p) ** (1 / p)
    return distance

def compute1_10(vector1, vector2):
    distances = []
    for p in range(1, 11):
        d = calculate_distances(vector1, vector2, p)
        distances.append(d)

        if p == 1:
            print(f"p = {p} (Manhattan) : {d:.4f}")
        elif p == 2:
            print(f"p = {p} (Euclidean) : {d:.4f}")
        else:
            print(f"p = {p}              : {d:.4f}")
    return distances
def inbuilt(vector1,vector2):
    #inbuilt functions to calculate euclidean and manhattan distance
    distances=[]
    for i in range (1,11):
        minkowski_distance = (minkowski(vector1, vector2,i))
        distances.append(minkowski_distance)
    return distances
    
def plot_manual(distances):
    import matplotlib.pyplot as plt
    p_values = list(range(1, 11))
    plt.figure(figsize=(8, 5))

    plt.plot(p_values, distances,
         marker='o',
         linewidth=2)

    plt.title("Minkowski Distance for Different p Values")
    plt.xlabel("p value")
    plt.ylabel("Distance")
    plt.xticks(p_values)
    plt.grid(True)

# Highlight Manhattan and Euclidean
    plt.scatter(1, distances[0], s=80, label="Manhattan (p=1)")
    plt.scatter(2, distances[1], s=80, label="Euclidean (p=2)")

    plt.legend()

    plt.show()

print("Minkowski Distance (p = 1 to 10)")
print("-" * 40)

vector1, vector2 = extract_numeric(load())
distances = compute1_10(vector1, vector2)
plot_manual(distances)
plot_manual(inbuilt(vector1,vector2))