import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def load_dataset():
    X_train = np.load("features/X_train_page.npy")
    return X_train
def create_dataframe(X_train):
    feature_names = [f"Feature_{i+1}"for i in range(X_train.shape[1])]
    df = pd.DataFrame(X_train,columns=feature_names)
    return df
def select_features_for_heatmap(df):
    heatmap_data = df.iloc[:, :50]
    return heatmap_data
def calculate_correlation(df):
    correlation_matrix = df.corr()
    return correlation_matrix
def plot_heatmap(correlation_matrix):
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        center=0
    )
    plt.title("Feature Correlation Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()
X_train=load_dataset()
df=create_dataframe(X_train)
print(X_train)
heat_data=select_features_for_heatmap(df)
cor_matrix=calculate_correlation(heat_data)
plot_heatmap(cor_matrix)