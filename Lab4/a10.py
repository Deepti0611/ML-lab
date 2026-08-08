import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def load_data(): # load data 
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign",na_values=["?"]) #derive data from excel sheet 
    return df


def choose_column(df):
# Select a feature
    feature = df["Income"]
    return feature
def plot_histogram(feature):
# Plot histogram and get histogram data
    frequency, bin_edges, patches = plt.hist(
    feature,
    bins=10,
    edgecolor='black'
)

    plt.title("Histogram of Income")
    plt.xlabel("Income")
    plt.ylabel("Frequency")
    plt.show()

    return frequency, bin_edges, patches
def mean(frequency, bin_edges):
    midpoints = []
    for i in range(len(bin_edges) - 1):
        midpoints.append((bin_edges[i] + bin_edges[i + 1]) / 2)

    mean = sum(f * m for f, m in zip(frequency, midpoints)) / sum(frequency)
    return mean,midpoints
def variance(frequency, midpoints, mean):
    variance = sum(f * (m - mean) ** 2
               for f, m in zip(frequency, midpoints)) / sum(frequency)

    return variance



df=load_data()
feature=choose_column(df)
frequency, bin_edges, patches = plot_histogram(feature)
mean,midpoints = mean(frequency, bin_edges)
variance = variance(frequency, midpoints, mean)
print("Mean of income column is:", mean)
print("Variance of income column is:", variance)
