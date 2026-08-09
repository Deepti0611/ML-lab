import time
import tracemalloc

from kmeans import (
    load_data,
    choose_columns,
    standardize_data,
    wcss_val,
    k_means
)


# Start performance measurement
start_time = time.perf_counter()

tracemalloc.start()


# Load
df = load_data()

# Feature selection
X = choose_columns(df)

X = X.dropna()

df = df.loc[X.index]

# Standardization
X_scaled, scaler = standardize_data(X)

# Elbow method
optimal_k = wcss_val(X_scaled)

# K-Means
model, centers, result_df = k_means(
    X,
    X_scaled,
    df,
    optimal_k,
    scaler
)


# End measurements
end_time = time.perf_counter()

current, peak = tracemalloc.get_traced_memory()

tracemalloc.stop()


# Results
execution_time = end_time - start_time

print("\n------------------------------")
print("PERFORMANCE TEST RESULTS")
print("------------------------------")

print("Execution Time :",
      round(execution_time, 4),
      "seconds")

print("Peak Memory    :",
      round(peak / 1024, 2),
      "KB")