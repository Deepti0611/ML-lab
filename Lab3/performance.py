import time

from kmeans import (
    load_data,
    choose_columns,
    k_means
)


# Start timer
start = time.perf_counter()


# Load data
df = load_data()


# Select features
points = choose_columns(df)


# Run K-Means
c1, c2, c3, centroid1, centroid2, centroid3 = k_means(points)


# Stop timer
end = time.perf_counter()


execution_time = end - start


print("-----------------------------")
print("PERFORMANCE TEST")
print("-----------------------------")

print("Number of data points:",
      len(points))

print("Number of features:",
      points.shape[1])

print("Execution Time:",
      round(execution_time, 6),
      "seconds")