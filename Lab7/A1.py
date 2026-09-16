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
def gini(prob):
    gini=1-np.sum(prob**2) # calculate the gini index of the probability distribution
    return gini
    
df=load_data() # load the data
print(df) # print the selected samples
prob=probability(df) # calculate the probability of each person
print(prob) # print the probabilities
ent=entropy(prob) # calculate the entropy of the probability distribution
print(ent) # print the entropy
gin=gini(prob) # calculate the gini index of the probability distribution
print(gin) # print the gini index