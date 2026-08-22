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
from sklearn.neighbors import KNeighborsClassifier


def create_weighted_knn(k):
    model = KNeighborsClassifier(
        n_neighbors=k,
        weights="distance"
    )

    return model


def fit_weighted_knn(model, X_train, y_train):

    model.fit(
        X_train,
        y_train
    )

    return model


def predict_weighted_knn(model, X_test):

    predictions = model.predict(
        X_test
    )

    return predictions


def score_weighted_knn(model, X_test, y_test):

    accuracy = model.score(
        X_test,
        y_test
    )

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
# Create weighted KNN model with k = 3
model = create_weighted_knn(3)

# Fit the model
model = fit_weighted_knn(
    model,
    X_train_processed,
    y_train
)

# Predict
predictions = predict_weighted_knn(
    model,
    X_test_processed
)

print("\nLibrary-based Weighted KNN Predictions:")
print(predictions)

print("\nActual:")
print(y_test)

# Calculate accuracy
accuracy = score_weighted_knn(
    model,
    X_test_processed,
    y_test
)

print("\nLibrary-based Weighted KNN Accuracy:")
print(accuracy)

print("\nLibrary-based Weighted KNN Accuracy (%):")
print(accuracy * 100)