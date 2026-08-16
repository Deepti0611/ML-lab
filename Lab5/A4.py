import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
def load_data():
    # Load data from CSV file
    df = pd.read_csv("features.csv")
    return df
def select_class(df):
    # Select only two classes from person_id
    selected_df = df[df["person_id"].isin(["A", "B"])].copy()

    return selected_df
def target_feature(df):
    # Separate features and target
    X = df.drop(columns=["person_id", "image_name"])
    Y = df["person_id"]
    return X, Y
def split_data(X, Y, test_size=0.3):
    # Divide data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=test_size,random_state=42,stratify=Y )
    return X_train, X_test, Y_train, Y_test
def KNN(X_train, Y_train,X_test,Y_test):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, Y_train)
    score=neigh.score(X_test,Y_test)
    predicted=neigh.predict(X_test)


    return score ,predicted
df = load_data()
# Select two classes
df = select_class(df)
print("Selected dataset:")
print(df)
# Separate features and target
X, Y = target_feature(df)
print("\nFeatures:")
print(X)
print("\nTarget:")
print(Y)
# Split into training and testing data
X_train, X_test, Y_train, Y_test = split_data(X, Y,test_size=0.3)
print("\nX_train:")
print(X_train)
print("\nY_train:")
print(Y_train)
print("\nX_test:")
print(X_test)
print("\nY_test:")
print(Y_test)
score,predicted=KNN(X_train,Y_train,X_test,Y_test)
print("\n accuracy score is:",score)
print("\n preiction is :",predicted)