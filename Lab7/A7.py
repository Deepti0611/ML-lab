import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay
def load_data(): # load data
    df = pd.read_csv("features.csv") #derive data from csv file
    return df
# from the previous experiment it was found that the highest info gain was obtained by bottom_left_B_var → IG ≈ 1.441 and bottom_right_B_mean → IG ≈ 1.433
#hence only these two features will be used to build the decision tree
def feature_selection(df):
    X = df.drop(columns=['person_id', 'image_name'])
    y = df['person_id']
    # Select the two features obtained from Information Gain
    selected_features = ['bottom_left_B_var', 'bottom_right_B_mean']
    X = X[selected_features]
    return X, y, selected_features
def decision_tree_classifier(X, y):
    clf = DecisionTreeClassifier(random_state=1)
    clf.fit(X, y)
    return clf
def plot(clf, X, y, selected_features):
    fig, ax =plt.subplots(figsize=(10, 7))

    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        response_method="predict",
        ax=ax,
        xlabel=selected_features[0],
        ylabel=selected_features[1],
        alpha=0.5
    )

    # Plot the training points
    scatter = disp.ax_.scatter(
        X.iloc[:, 0],
        X.iloc[:, 1],
        c=pd.factorize(y)[0],
        cmap=ListedColormap(disp.multiclass_colors_),
        edgecolor="black",
        s=30
    )

    plt.title("Decision Surface of Decision Tree")
    plt.tight_layout()

    plt.show() 
df=load_data() # load the data
X, y, selected_features = feature_selection(df) # feature selection
clf = decision_tree_classifier(X, y) # decision tree classifier
plot(clf, X, y, selected_features) # plot the decision surface
