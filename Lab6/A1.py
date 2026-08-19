import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
def load_data(): # load data 
    df = pd.read_csv("features.csv") #derive data from csv file 
    return df
def select_class(df):
    selected_classes = df["person_id"].unique()[:2]
    df_two_classes = df[df["person_id"].isin(selected_classes)]
    return df_two_classes
def target_feature(df):
    # Separate features and target
    X = df.drop(columns=["person_id", "image_name"])
    Y = df["person_id"]
    return X, Y
def splitdata(x,y):
    X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)
    return X_train,X_test,y_train,y_test
def encode_features(X):
    """
    Common encoding module for numerical and categorical features.

    Numerical columns:
        Standardized using StandardScaler

    Categorical columns:
        Encoded using OneHotEncoder
    """

    # Identify numerical and categorical columns
    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    # Numerical preprocessing
    numerical_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    # Categorical preprocessing
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ))
        ]
    )

    # Combine both
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_transformer, numerical_columns),
            ("categorical", categorical_transformer, categorical_columns)
        ]
    )

    # Transform the data
    X_encoded = preprocessor.fit_transform(X)

    return X_encoded, preprocessor
def create_preprocessor(X):

    # Identify column types
    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    # Numerical:
    # 1. Missing values -> median
    # 2. Scale the values
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    # Categorical:
    # 1. Missing values -> most frequent (mode)
    # 2. One-hot encode
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ))
        ]
    )

    # Combine both
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_transformer, numerical_columns),
            ("categorical", categorical_transformer, categorical_columns)
        ]
    )

    return preprocessor
def distance(vector_1,vector_2):
    distance = np.linalg.norm(vector_1 - vector_2)
    return distance
def bubble_sort(distances):
    arr = distances.copy()

    n = len(arr)

    for i in range(n):
        swapped = False

        for j in range(0, n - i - 1):

            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True

        if not swapped:
            break

    return arr


def insertion_sort(distances):
    arr = distances.copy()

    for i in range(1, len(arr)):

        key = arr[i]
        j = i - 1

        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1

        arr[j + 1] = key

    return arr


def merge_sort(distances):

    if len(distances) <= 1:
        return distances.copy()

    mid = len(distances) // 2

    left = merge_sort(distances[:mid])
    right = merge_sort(distances[mid:])

    return merge(left, right)


def merge(left, right):

    result = []

    i = 0
    j = 0

    while i < len(left) and j < len(right):

        if left[i] <= right[j]:
            result.append(left[i])
            i += 1

        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])

    return result



class KNN:

    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        Store training data and training labels.
        """

        self.X_train = X_train
        self.y_train = y_train

        return self

    def predict_one(self, test_vector):
        """
        Predict the class of one test vector.
        """

        distances = []

        # Calculate distance between test vector
        # and every training vector
        for i in range(len(self.X_train)):

            dist = distance(
                test_vector,
                self.X_train[i]
            )

            distances.append(
                (dist, self.y_train[i])
            )

        # Sort distances
        sorted_distances = merge_sort(distances)

        # Select K nearest neighbours
        k_nearest = sorted_distances[:self.k]

        # Get labels
        labels = []

        for dist, label in k_nearest:
            labels.append(label)

        # Majority voting
        prediction = max(
            set(labels),
            key=labels.count
        )

        return prediction

    def predict(self, X_test):
        """
        Predict classes for all test vectors.
        """

        predictions = []

        for test_vector in X_test:

            prediction = self.predict_one(
                test_vector
            )

            predictions.append(prediction)

        return predictions
def accuracy_score(y_true, y_pred):

    if len(y_true) != len(y_pred):
        raise ValueError(
            "Actual and predicted values must have the same length"
        )

    correct = 0

    for actual, predicted in zip(y_true, y_pred):

        if actual == predicted:
            correct += 1

    accuracy = correct / len(y_true)

    return accuracy

df = load_data()

# Select two classes
df = select_class(df)

# Separate features and target
x, y = target_feature(df)

print("\nFeatures:")
print(x)

print("\nTarget:")
print(y)

X_train, X_test, y_train, y_test = splitdata(x, y)

print("\nX_train:")
print(X_train)

print("\nX_test:")
print(X_test)

print("\ny_train:")
print(y_train)

print("\ny_test:")
print(y_test)

preprocessor = create_preprocessor(X_train)


# Fit preprocessor ONLY on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Use the same fitted preprocessor on test data
X_test_processed = preprocessor.transform(X_test)


print("\nProcessed X_train:")
print(X_train_processed)

print("\nProcessed X_test:")
print(X_test_processed)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

model = KNN(k=3)

model.fit(
    X_train_processed,
    y_train
)

predictions = model.predict(
    X_test_processed
)

print("\nPredicted classes:")
print(predictions)

print("\nActual classes:")
print(y_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nAccuracy:")
print(accuracy)
