
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def load_dataset():
    df = pd.read_csv("features.csv")
    X = df.drop(columns=["person_id", "image_name"])
    y = LabelEncoder().fit_transform(df["person_id"])
    return X, y
def split_dataset(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
def svm_classifier(X_train, X_test, y_train, y_test):
    model = SVC()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {"Accuracy": accuracy_score(y_test, predictions),"Precision": precision_score(y_test, predictions, average="macro", zero_division=0),"Recall": recall_score(y_test, predictions, average="macro", zero_division=0),"F1 Score": f1_score(y_test, predictions, average="macro", zero_division=0)}
def decision_tree_classifier(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {"Accuracy": accuracy_score(y_test, predictions),"Precision": precision_score(y_test, predictions, average="macro", zero_division=0),"Recall": recall_score(y_test, predictions, average="macro", zero_division=0),"F1 Score": f1_score(y_test, predictions, average="macro", zero_division=0)}
def random_forest_classifier(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {"Accuracy": accuracy_score(y_test, predictions),"Precision": precision_score(y_test, predictions, average="macro", zero_division=0),"Recall": recall_score(y_test, predictions, average="macro", zero_division=0),"F1 Score": f1_score(y_test, predictions, average="macro", zero_division=0)}
def catboost_classifier(X_train, X_test, y_train, y_test):
    model = CatBoostClassifier(iterations=100, verbose=0, random_seed=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test).ravel()
    return {"Accuracy": accuracy_score(y_test, predictions),"Precision": precision_score(y_test, predictions, average="macro", zero_division=0),"Recall": recall_score(y_test, predictions, average="macro", zero_division=0),"F1 Score": f1_score(y_test, predictions, average="macro", zero_division=0)}
def adaboost_classifier(X_train, X_test, y_train, y_test):
    model = AdaBoostClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {"Accuracy": accuracy_score(y_test, predictions),"Precision": precision_score(y_test, predictions, average="macro", zero_division=0),"Recall": recall_score(y_test, predictions, average="macro", zero_division=0),"F1 Score": f1_score(y_test, predictions, average="macro", zero_division=0)}
def xgboost_classifier(X_train, X_test, y_train, y_test):
    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {"Accuracy": accuracy_score(y_test, predictions),"Precision": precision_score(y_test, predictions, average="macro", zero_division=0),"Recall": recall_score(y_test, predictions, average="macro", zero_division=0),"F1 Score": f1_score(y_test, predictions, average="macro", zero_division=0)}
def naive_bayes_classifier(X_train, X_test, y_train, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {"Accuracy": accuracy_score(y_test, predictions),"Precision": precision_score(y_test, predictions, average="macro", zero_division=0),"Recall": recall_score(y_test, predictions, average="macro", zero_division=0),"F1 Score": f1_score(y_test, predictions, average="macro", zero_division=0)}

X, y = load_dataset()
X_train, X_test, y_train, y_test = split_dataset(X, y)
svm_results = svm_classifier(X_train, X_test, y_train, y_test)
dt_results = decision_tree_classifier(X_train, X_test, y_train, y_test)
rf_results = random_forest_classifier(X_train, X_test, y_train, y_test)
cat_results = catboost_classifier(X_train, X_test, y_train, y_test)
ada_results = adaboost_classifier(X_train, X_test, y_train, y_test)
xgb_results = xgboost_classifier(X_train, X_test, y_train, y_test)
nb_results = naive_bayes_classifier(X_train, X_test, y_train, y_test)
results = {"SVM": svm_results,"Decision Tree": dt_results,"Random Forest": rf_results,"CatBoost": cat_results,"AdaBoost": ada_results,"XGBoost": xgb_results,"Naive Bayes": nb_results}
results_df = pd.DataFrame(results)
print("Classifier Comparison Results")
print(results_df)

