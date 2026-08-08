import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
def load_data(): # load data 
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign",na_values=["?"]) #derive data from excel sheet 
    return df
def choose_columns(df):
    X = df.select_dtypes(include=[np.number])
    return X
def standardize_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
def wcss_val(X_scaled):
    wcss = []
    K=range(1, 11)
    for k in range(1, 11):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_scaled)
        wcss.append(model.inertia_)

    # Coordinates of first and last points
    x1, y1 = 1, wcss[0]
    x2, y2 = 10, wcss[-1]

    distances = []

# Distance of each point from the line
    for i, y0 in zip(K, wcss):

        numerator = abs((y2-y1)*i - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = np.sqrt((y2-y1)**2 + (x2-x1)**2)

        distances.append(numerator/denominator)

# Elbow point
        optimal_k = K[np.argmax(distances)]

        print("Optimal K =", optimal_k)

# Plot
    plt.figure(figsize=(8,5))
    plt.plot(K, wcss, marker='o')
    plt.scatter(optimal_k,
            wcss[optimal_k-1],
            color='red',
            s=120,
            label=f'Elbow = {optimal_k}')

    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    plt.legend()
    plt.grid(True)

    plt.show()
    return optimal_k
def k_means(X, X_scaled, df, optimal_k, scaler):
    k = optimal_k
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
)

    kmeans.fit(X_scaled)

# Labels
    df["Cluster"] = kmeans.labels_

    print(df[["Cluster"]].head())

# Cluster Centers
    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    print("\nCluster Centers")
    print(centers)

# Plot
    plt.figure(figsize=(8,6))

    plt.scatter(
        X_scaled[:,0],
        X_scaled[:,1],
        c=kmeans.labels_,
        cmap='viridis'
)

    plt.scatter(
        kmeans.cluster_centers_[:,0],
        kmeans.cluster_centers_[:,1],
        c='red',
        marker='X',
        s=200,
        label='Centroids'
)

    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])

    plt.title("K-Means Clustering")

    plt.legend()

    plt.show()
df=load_data()
X=choose_columns(df)
X = X.dropna()
df = df.loc[X.index]
scaled_X, scaler = standardize_data(X)
optimal_k = wcss_val(scaled_X)
k_means(X, scaled_X, df, optimal_k, scaler)