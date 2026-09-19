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
        df[f'{colname}_bin'] = pd.qcut(df[colname], q=K, labels=labels,duplicates='drop')
    else:
        raise ValueError("Invalid binning type. Choose 'equal_width' or 'equal_frequency'.")
    return df
def info_gain(ent,df,bin_type='equal_width',K=5):
    max_gain=-1
    best_col=None
    weighted_entropy=0
    features = df.select_dtypes(include=np.number).columns
    for col in features:
        df=binning(df, col, bin_type, K) # perform binning
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
        print(f'{col}: {gain}')
        if gain>max_gain:
            max_gain=gain
            best_col=col
    return best_col, max_gain
    
df = load_data()
prob = probability(df)
initial_entropy = entropy(prob)
print("Initial Entropy:", initial_entropy)
best_col, max_gain = info_gain(initial_entropy, df)
print("Default binning:")
print("Best column:", best_col)
print("Maximum Information Gain:", max_gain)

best_col, max_gain = info_gain(initial_entropy,df,'equal_frequency',5)
print("Frequency binning:")
print("Best column:", best_col)
print("Maximum Information Gain:", max_gain)