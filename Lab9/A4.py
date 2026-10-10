
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def load_dataset():
    df = pd.read_csv("features.csv")
    X = df.drop(columns=["person_id", "image_name"])
    y = df["person_id"]
    return X, y
def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test
def create_pipeline():
    pipeline = Pipeline([("scaler", StandardScaler()),("classifier", SVC())])
    return pipeline
def run_pipeline(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    results = {"Accuracy": accuracy_score(y_test, predictions),"Precision": precision_score(y_test, predictions, average="macro", zero_division=0),
                "Recall": recall_score(y_test, predictions, average="macro", zero_division=0),
                "F1 Score": f1_score(y_test, predictions, average="macro", zero_division=0)}
    return results, predictions

X, y = load_dataset()
X_train, X_test, y_train, y_test = split_dataset(X, y)
pipeline = create_pipeline()
results, predictions = run_pipeline(pipeline, X_train, X_test, y_train, y_test)
print("Pipeline:", pipeline)
print("\nPerformance Metrics:")
print(results)
print("\nPredictions:", predictions)
print("Actual Values:", y_test.to_numpy())

