import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
def load_dataset():
    X_train = np.load("features/X_train_page.npy")
    y_train = np.load("features/y_train_page.npy")
    return X_train, y_train
def tune_perceptron(X, y):
    perceptron = Perceptron(random_state=0)
    distributions = {"alpha": uniform(loc=0.0001, scale=0.01), "tol": uniform(loc=0.0001, scale=0.01), "max_iter": [100, 500, 1000, 2000]}
    clf = RandomizedSearchCV(perceptron,distributions,n_iter=10,cv=4,random_state=0)
    search = clf.fit(X, y)
    return search
def tune_mlp(X, y):
    mlp = MLPClassifier(random_state=1)
    distributions = {"hidden_layer_sizes": [(10,), (15,), (20,), (50,), (10, 10)],"activation": ["relu", "tanh", "logistic"],"solver": ["adam", "lbfgs"], "alpha": uniform(loc=0.0001, scale=0.01),"max_iter": [1000,2000,5000]}
    clf = RandomizedSearchCV( mlp,distributions, n_iter=10,cv=4,random_state=0)
    search = clf.fit(X, y)
    return search
X_train, y_train = load_dataset()
perceptron_search = tune_perceptron(X_train, y_train)
mlp_search = tune_mlp(X_train, y_train)
print("Perceptron Best Parameters:", perceptron_search.best_params_)
print("Perceptron Best CV Accuracy:", perceptron_search.best_score_)
print("Perceptron Best Model:", perceptron_search.best_estimator_)
print("\nMLP Best Parameters:", mlp_search.best_params_)
print("MLP Best CV Accuracy:", mlp_search.best_score_)
print("MLP Best Model:", mlp_search.best_estimator_)
