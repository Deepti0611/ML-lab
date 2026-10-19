import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier,AdaBoostClassifier)
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score)
def load_dataset():
    X_train = np.load("features/X_train_page.npy")
    y_train = np.load("features/y_train_page.npy")
    X_test = np.load("features/X_test_page.npy")
    y_test = np.load("features/y_test_page.npy")
    return X_train, X_test, y_train, y_test
def apply_pca(X_train):
    pca = PCA(n_components=0.99)
    X_train_pca = pca.fit_transform(X_train)
    return pca, X_train_pca
def transform_test_data(pca, X_test):
    X_test_pca = pca.transform(X_test)
    return X_test_pca
def create_classification_models():
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "SVM": SVC(kernel="rbf",random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100,random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100,random_state=42)
    }
    return models
def evaluate_models(models, X_train, y_train, X_test, y_test):
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score( y_test,y_pred,average="weighted",zero_division=0)
        recall = recall_score(y_test,y_pred,average="weighted",zero_division=0)
        f1 = f1_score(y_test,y_pred,average="weighted",zero_division=0)
        results.append([name,accuracy,precision,recall,f1])
    return results
X_train, X_test, y_train, y_test = load_dataset()
print("Original X_train shape:", X_train.shape)
print("Original X_test shape:", X_test.shape)
# Apply PCA
pca, X_train_pca = apply_pca(X_train)
# Transform test data
X_test_pca = transform_test_data(pca,X_test)

print("\nNumber of PCA components:", pca.n_components_)
print(
    "Cumulative explained variance:",
    np.sum(pca.explained_variance_ratio_)
)
print(
    "PCA X_train shape:",
    X_train_pca.shape
)
print(
    "PCA X_test shape:",
    X_test_pca.shape
)
models = create_classification_models()
results = evaluate_models(models,X_train_pca,y_train,X_test_pca,y_test)
results_df = pd.DataFrame(results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)
print("\nCLASSIFICATION RESULTS WITH PCA")
print(results_df.to_string(index=False))