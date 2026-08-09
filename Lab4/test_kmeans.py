import unittest
import pandas as pd
import numpy as np

from kmeans import (
    load_data,
    choose_columns,
    standardize_data,
    wcss_val,
    k_means
)


class TestKMeans(unittest.TestCase):

    def test_load_data(self):

        df = load_data()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)


    def test_choose_columns(self):

        df = load_data()

        X = choose_columns(df)

        self.assertTrue(
            all(np.issubdtype(dtype, np.number) for dtype in X.dtypes)
        )


    def test_standardize_shape(self):

        df = load_data()

        X = choose_columns(df)

        X = X.dropna()

        scaled_X, scaler = standardize_data(X)

        self.assertEqual(
            scaled_X.shape,
            X.shape
        )


    def test_standardize_mean(self):

        df = load_data()

        X = choose_columns(df)

        X = X.dropna()

        scaled_X, scaler = standardize_data(X)

        means = np.mean(scaled_X, axis=0)

        self.assertTrue(
            np.allclose(means, 0, atol=1e-6)
        )


    def test_standardize_std(self):

        df = load_data()

        X = choose_columns(df)
        X = X.dropna()

        scaled, scaler = standardize_data(X)

        std = np.std(scaled, axis=0)

        print("\nStandard deviations:")
        print(std)

        print("\nColumns:")
        print(X.columns)

        self.assertTrue(
        np.allclose(std[std != 0], 1, atol=1e-6)
)


    def test_optimal_k(self):

        df = load_data()

        X = choose_columns(df)

        X = X.dropna()

        scaled_X, scaler = standardize_data(X)

        k = wcss_val(scaled_X)

        self.assertTrue(
            1 <= k <= 10
        )


    def test_cluster_labels(self):

        df = load_data()

        X = choose_columns(df)

        X = X.dropna()

        df = df.loc[X.index]

        scaled_X, scaler = standardize_data(X)

        k = wcss_val(scaled_X)

        model = k_means(
            X,
            scaled_X,
            df,
            k,
            scaler
        )

        self.assertEqual(
            len(model.labels_),
            len(X)
        )


if __name__ == "__main__":
    unittest.main()