import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
def load_data(): # load data 
    df = pd.read_csv("features.csv") #derive data from csv file 
    return df
'''def target_feature(df): # separate target and features
    X=df.drop(columns=["person_id","image_name"])
    Y=df["person_id"]
    return X, Y'''
    
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

def encoding(X):   
# Identify categorical columns
    categorical_columns = X.select_dtypes(
    include=["object"]
    ).columns.tolist()
    if(categorical_columns):
        # Create a ColumnTransformer with OneHotEncoder for categorical columns
        column_transformer = ColumnTransformer(
            transformers=[
                ("onehot", OneHotEncoder(), categorical_columns)
            ],
            )
        # Fit and transform the data
        X_encoded = column_transformer.fit_transform(X)
        return X_encoded
    else:
        return X
def missing(X,method): # handle missing values
    X=pd.DataFrame(X).copy()
    for col in X.columns:
        if X[col].isnull().any():
            if method == "mean":
                X[col]=X[col].fillna(X[col].mean())
            elif method == "median":
                X[col]=X[col].fillna(X[col].median())
            elif method == "mode":
                X[col]=X[col].fillna(X[col].mode()[0])
    return X
def eucledean(vec1,vec2): # calculate eucledean distance
    return np.sqrt(np.sum((vec1-vec2)**2))
def KNN_dist(train,test,train_labels): # calculate distance between train and test data
    distances=[]
    train=np.array(train)
    test=np.array(test)
    train_labels=np.array(train_labels)
    for i in range(len(train)):
        dist=eucledean(train[i],test)
        distances.append([dist, train_labels[i]])
    return distances
def sort_bubble(distances):
    n = len(distances)
    distances=distances.copy()
    for i in range(n):
        for j in range(0, n-i-1):
            if distances[j][0] > distances[j+1][0]:
                distances[j], distances[j+1] = distances[j+1], distances[j]
    return distances
def sort_selection(distances):
    n = len(distances)
    distances=distances.copy()
    for i in range(n):
        min = i
        for j in range(i+1, n):
            if distances[j][0] < distances[min][0]:
                min = j
        distances[i], distances[min] = distances[min], distances[i]
    return distances
def sort_insertion(distances):
    n = len(distances)
    for i in range(1, n):
        key = distances[i]
        j = i-1
        while j >= 0 and key[0] < distances[j][0]:
            distances[j + 1] = distances[j]
            j -= 1
        distances[j + 1] = key
    return distances
def identify_k(distances,k): # identify k nearest neighbours
    return distances[:k]
def majority(neighbours):
    labels = [item[1] for item in neighbours]
    class_counts = {}
    for label in labels:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    maximum = max(class_counts.values())
    majority_labels = []
    for label in class_counts:

        if class_counts[label] == maximum:
            majority_labels.append(label)

    # No tie
    if len(majority_labels) == 1:
        return majority_labels[0]

    # Tie:
    # Since neighbours are already sorted by distance,
    # choose the class of the closest tied class.
    for item in neighbours:

        if item[1] in majority_labels:
            return item[1]
def fit(X,Y):
    X_train,X_test,Y_train,Y_test=split_data(X,Y,test_size=0.3)
    return X_train,X_test,Y_train,Y_test
    
def predict(X_train,Y_train , X_test,k):#prediction for all X_test
    predictions = []
    for test_vector in X_test.values:
        distances = KNN_dist(X_train,test_vector,Y_train)
        distances = sort_bubble(distances)
        neighbours = identify_k(distances, k)
        prediction = majority(neighbours)
        predictions.append(prediction)
    return predictions           
def score(prediction,Y_test):
    sum=0
    for i in range(len(prediction)):
        if prediction[i]==Y_test.iloc[i]:
            sum=sum+1
    score=sum/len(prediction)
    return score

def comparison(X_train,X_test,Y_train,Y_test):
    my_accuracies = []
    sklearn_accuracies = [] 
    kval=[1,2,3,4,5,6,7]
    for k in  kval:
        predictions = predict(X_train,Y_train,X_test,k)

        accuracy = score(predictions,Y_test )

        my_accuracies.append(accuracy)
        model = KNeighborsClassifier(n_neighbors=k)

        model.fit(X_train,Y_train)

        sklearn_accuracy = model.score(X_test,Y_test)

        sklearn_accuracies.append(sklearn_accuracy)
    results = pd.DataFrame({
    "My KNN Accuracy": my_accuracies,
    "Sklearn KNN Accuracy": sklearn_accuracies
})
    return results,my_accuracies,sklearn_accuracies
def plot(my_accuracies,sklearn_accuracies,kval):
    plt.plot(
    kval,
    my_accuracies,
    marker="o",
    label="My KNN"
)

    plt.plot(
        kval,
        sklearn_accuracies,
        marker="o",
        label="Sklearn KNN"
)

    plt.xlabel("Value of k")
    plt.ylabel("Accuracy")
    plt.title("Comparison of Custom KNN and Sklearn KNN")
    plt.show()
      
df = load_data()
# Select two classes
df = select_class(df)
# Separate features and target
X, Y = target_feature(df)
# Handle missing values
X = missing(X, "mean")
X_train, X_test, Y_train, Y_test = fit(X,Y)

result,my_accuracies,sklearn_accuracies=comparison(X_train,X_test,Y_train,Y_test)
print("\nComparison of Custom KNN and Sklearn KNN:")
print(result)
plot(my_accuracies,sklearn_accuracies,[1,2,3,4,5,6,7])