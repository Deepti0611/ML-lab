import numpy as np
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier,AdaBoostClassifier,StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score)
def load_dataset():
    X_train = np.load("features/X_train_page.npy")
    y_train = np.load("features/y_train_page.npy")
    X_test = np.load("features/X_test_page.npy")
    y_test = np.load("features/y_test_page.npy")
    return X_train, X_test, y_train, y_test
def create_base_models(): #base model
    knn = KNeighborsClassifier(n_neighbors=3)
    svm = SVC(kernel="rbf",probability=True,random_state=42)
    rf = RandomForestClassifier(n_estimators=100,random_state=42)
    adaboost = AdaBoostClassifier(n_estimators=100,random_state=42)
    estimators = [("knn", knn),("svm", svm),("rf", rf),("adaboost", adaboost)]
    return estimators
def create_final_estimators():# meta model
    final_estimators = {
        "Logistic Regression": LogisticRegression(max_iter=1000,random_state=42),
        "SVM": SVC(probability=True,random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100,random_state=42)
    }
    return final_estimators
def create_stacking_model(estimators, final_estimator): #stacking classifer 
    model = StackingClassifier(estimators=estimators,final_estimator=final_estimator,cv=4)
    return model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test,y_pred, average="weighted",zero_division=0)
    recall = recall_score(y_test,y_pred,average="weighted",zero_division=0)
    f1 = f1_score( y_test,y_pred, average="weighted", zero_division=0)
    # weighted average because this is a multiclass problem with 20 classes
    return accuracy, precision, recall, f1
# Main program
X_train, X_test, y_train, y_test = load_dataset()
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
estimators = create_base_models()
final_estimators = create_final_estimators()
results = []
for name, final_estimator in final_estimators.items():
    print("\nTraining Stacking Classifier with:", name)
    model = create_stacking_model(estimators,final_estimator)
    model.fit(X_train,y_train)
    accuracy, precision, recall, f1 = evaluate_model(model,X_test,y_test)
    results.append([name,accuracy,precision,recall,f1])
print("\n")
print("STACKING CLASSIFIER RESULTS")

print(
    f"{'Final Estimator':<25}"
    f"{'Accuracy':<15}"
    f"{'Precision':<15}"
    f"{'Recall':<15}"
    f"{'F1 Score':<15}"
)
for result in results:
    print(
        f"{result[0]:<25}"
        f"{result[1]:<15.4f}"
        f"{result[2]:<15.4f}"
        f"{result[3]:<15.4f}"
        f"{result[4]:<15.4f}"
    )