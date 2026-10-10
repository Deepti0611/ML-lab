
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from lime.lime_tabular import LimeTabularExplainer
def load_dataset():
    df = pd.read_csv("features.csv")
    X = df.drop(columns=["person_id", "image_name"])
    y = df["person_id"]
    return X, y
def split_dataset(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
def create_pipeline():
    pipeline = Pipeline([("scaler", StandardScaler()),("classifier", SVC(probability=True))])
    return pipeline
def train_pipeline(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline
def explain_prediction(pipeline, X_train, X_test, y_test, index=0):
    explainer = LimeTabularExplainer(training_data=X_train.to_numpy(),feature_names=X_train.columns.tolist(),class_names=pipeline.named_steps["classifier"].classes_.tolist(),mode="classification")
    sample = X_test.iloc[index].to_numpy()
    prediction = pipeline.predict(X_test.iloc[[index]])[0]
    actual = y_test.iloc[index]
    class_index = list(pipeline.named_steps["classifier"].classes_).index(prediction)
    explanation = explainer.explain_instance(data_row=sample,predict_fn=lambda data: pipeline.predict_proba(pd.DataFrame(data, columns=X_train.columns)),num_features=10,labels=[class_index])
    return explanation, prediction, actual, class_index
X, y = load_dataset()
X_train, X_test, y_train, y_test = split_dataset(X, y)
pipeline = create_pipeline()
pipeline = train_pipeline(pipeline, X_train, y_train)
explanation, prediction, actual, class_index = explain_prediction(pipeline, X_train, X_test, y_test, index=0)
print("Actual Class:", actual)
print("Predicted Class:", prediction)
print("\nLIME Explanation:")
print(explanation.as_list(label=class_index))


