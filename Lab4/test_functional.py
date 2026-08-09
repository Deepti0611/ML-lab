import unittest
import numpy as np

from kmeans import (
    load_data,
    choose_columns,
    standardize_data,
    wcss_val,
    k_means
)


class TestKMeansFunctionality(unittest.TestCase):

    def test_complete_workflow(self):

        # 1. Load data
        df = load_data()

        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)

        # 2. Select numeric features
        X = choose_columns(df)

        self.assertGreater(X.shape[1], 0)

        # 3. Remove missing values
        X = X.dropna()

        df = df.loc[X.index]

        self.assertEqual(len(X), len(df))

        # 4. Standardize
        X_scaled, scaler = standardize_data(X)

        self.assertEqual(
            X_scaled.shape,
            X.shape
        )

        # 5. Find optimal K
        optimal_k = wcss_val(X_scaled)

        self.assertTrue(
            1 <= optimal_k <= 10
        )

        # 6. Perform K-Means
        model, centers, result_df = k_means(
            X,
            X_scaled,
            df,
            optimal_k,
            scaler
        )

        # 7. Check cluster labels
        self.assertEqual(
            len(model.labels_),
            len(X)
        )

        # 8. Check number of clusters
        self.assertEqual(
            model.n_clusters,
            optimal_k
        )

        # 9. Check cluster centers
        self.assertEqual(
            centers.shape,
            (optimal_k, X.shape[1])
        )

        # 10. Check Cluster column
        self.assertIn(
            "Cluster",
            result_df.columns
        )

        print("\nFunctional Test Passed")


if __name__ == "__main__":
    unittest.main()