import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(): # load data
    df = pd.read_csv("features.csv") #derive data from csv file
    return df

def probability(df):
    count=df['person_id'].value_counts() # count the number of samples for each person
    prob=count/len(df) # calculate the probability of each person
    return prob
def entropy(prob):
    ent=np.sum(-prob*np.log2(prob)) # calculate the entropy of the probability distribution
    return ent

def binning(df,colname,bin_type='equal_width',K=5):
        # Number of bins
    labels = [f'Bin {i+1}' for i in range(K)]
    # Calculate the width of each bin
    if bin_type == 'equal_width':
        min_col = df[colname].min()
        max_col = df[colname].max()
        # Define bins using calculated width
        bins = np.linspace(min_col, max_col, num=K+1)
        #print(bins)
        # Create bin labels
        #print(labels)
        # Perform binning
        df[f'{colname}_bin'] = pd.cut(df[colname], bins=bins, labels=labels, include_lowest=True)
    elif bin_type == 'equal_frequency':
        # Perform equal frequency binning
        df[f'{colname}_bin'],bins = pd.qcut(df[colname], q=K, labels=labels,retbins=True,duplicates='drop')
    else:
        raise ValueError("Invalid binning type. Choose 'equal_width' or 'equal_frequency'.")
    return df,bins
def info_gain(ent,df,bin_type='equal_width',K=5,features=None):
    max_gain=-1
    best_col=None
    weighted_entropy=0
    if features is None:
        features = df.select_dtypes(include=np.number).columns.tolist()
    for col in features:
        df,bins=binning(df, col, bin_type, K) # perform binning
        binned_col = f'{col}_bin'
        weighted_entropy=0
        for i in range(1,K+1):
            current_bin=df[df[binned_col]==f'Bin {i}']
            if len(current_bin)==0:
                continue
            prob=probability(current_bin)
            bin_ent=entropy(prob)
            weight=len(current_bin)/len(df)
            weighted_entropy+=weight*bin_ent
        gain=ent-weighted_entropy
        if gain>max_gain:
            max_gain=gain
            best_col=col
    return best_col, max_gain,bins
class DecisionTree:
    def __init__(self, depth=0, max_depth=3, num_bins=5):
        self.children = {}
        self.feature = None
        self.bin_edges = None
        self.max_depth = max_depth
        self.num_bins = num_bins
        self.depth = depth
        self.target = None
    def train(self, X_train, y_train,features=None):
        # Stop if all samples belong to one class
        if features is None:
         features = X_train.columns.tolist()
        self.target = y_train.mode()[0]
        if y_train.nunique() == 1:
            self.target = y_train.iloc[0]
            return
        # Stop if maximum depth is reached
        if self.depth >= self.max_depth:
            self.target = y_train.mode()[0]
            return
        # Stop if there are no samples
        if len(y_train) == 0:
            return
        # Combine X and y
        data = X_train.copy()
        data['person_id'] = y_train.values
        # Find entropy of current node
        current_entropy = entropy(probability(data))
        # Find best feature
        best_feature, max_gain,bins = info_gain(current_entropy,data,features=features)
    # If no useful feature found
        if best_feature is None:
            self.target = y_train.mode()[0]
            return
        self.feature = best_feature
        # Bin the best feature
        data,bins = binning(data,best_feature,'equal_width',self.num_bins)
        self.bin_edges = bins
        bin_column = best_feature + '_bin'
        remaining_features = [
        feature for feature in features
        if feature != best_feature
    ]
        # Create child for every bin
        for bin_value in data[bin_column].dropna().unique():
            child_data = data[
                data[bin_column] == bin_value]
            if len(child_data) == 0:
                continue
            child_X = child_data[X_train.columns]
            child_y = child_data['person_id']
            # Create child node
            child = DecisionTree(depth=self.depth + 1,max_depth=self.max_depth,num_bins=self.num_bins)
            # Recursively train child
            child.train(child_X, child_y, features=remaining_features)
            self.children[str(bin_value)] = child
        # If no children were created
        if len(self.children) == 0:
            self.target = y_train.mode()[0] 
df = load_data()
X = df.drop(columns=['person_id', 'image_name'])
y = df['person_id']
features = X.columns.tolist()
tree = DecisionTree(max_depth=3, num_bins=5)
tree.train(X, y, features)
print("Decision Tree built successfully.")