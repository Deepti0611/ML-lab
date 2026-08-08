import pandas as pd
import numpy as np

def load():
    # Load the dataset
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign")
    return df
def matrix(df):
# Select only numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Convert to NumPy matrix
    matrix = numeric_df.to_numpy()
    return matrix,numeric_df
def mean(column):
    total = 0

    for value in column:
        total += value

    return total / len(column)
def variance(column):
    m = mean(column)

    total = 0

    for value in column:
        total += (value - m) ** 2

    return total / len(column)          # Population Variance
def standard_deviation(column):
    return variance(column) ** 0.5

df=load()
matrix, numeric_df = matrix(df)
print(matrix)
print("Shape:", matrix.shape)
print("Manual Calculations")
print("-"*60)

for i in range(matrix.shape[1]):

    column = matrix[:, i]

    print(f"\nColumn : {numeric_df.columns[i]}")
    print("Mean              :", mean(column))
    print("Variance          :", variance(column))
    print("Standard Deviation:", standard_deviation(column))

print("\nBuilt-in Calculations")
print("-"*60)

for i in range(matrix.shape[1]):

    column = matrix[:, i]

    print(f"\nColumn : {numeric_df.columns[i]}")
    print("Mean              :", np.mean(column))
    print("Variance          :", np.var(column))
    print("Standard Deviation:", np.std(column))
