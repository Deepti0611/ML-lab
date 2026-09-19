import numpy as np
# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics 
from sklearn.tree import export_graphviz
from io import StringIO  
from IPython.display import Image, display  
import pydotplus


def load_data(): # load data
    df = pd.read_csv("features.csv") #derive data from csv file
    return df
def feature_selection(df):
    x=df.drop(['person_id','image_name'],axis=1) # drop the person_id column
    y=df['person_id'] # select the person_id column as target variable
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) # split the data into training and testing sets
    return x_train, x_test, y_train, y_test
def decision_tree_classifier(x_train, y_train):
    clf = DecisionTreeClassifier() # Create Decision Tree classifer object
    clf = clf.fit(x_train,y_train) # Train Decision Tree Classifer

    return clf
def visualize_tree(clf, feature_cols):
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True,feature_names = feature_cols,class_names=clf.classes_.astype(str).tolist())
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('handwritten.png')
    return Image(graph.create_png())
df=load_data() # load the data
x_train, x_test, y_train, y_test=feature_selection(df) # feature selection
clf=decision_tree_classifier(x_train, y_train) # decision tree classifier
feature_cols = x_train.columns.tolist() # get the feature columns
tree_image = visualize_tree(clf, feature_cols) # visualize the decision tree
display(tree_image) # display the decision tree