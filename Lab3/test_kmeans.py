import unittest
import numpy as np
import pandas as pd

from kmeans import (
    load_data,
    choose_columns,
    k_means
)


class TestManualKMeans(unittest.TestCase):

    # -----------------------------
    # Test 1: Loading
    # -----------------------------

    def test_load_data(self):

        df = load_data()

        self.assertIsInstance(df, pd.DataFrame)

        self.assertFalse(df.empty)

        self.assertIn("Income", df.columns)

        self.assertIn("MntWines", df.columns)


    # -----------------------------
    # Test 2: Feature Selection
    # -----------------------------

    def test_choose_columns(self):

        df = load_data()

        points = choose_columns(df)

        self.assertIsInstance(points, np.ndarray)

        self.assertEqual(points.shape[1], 2)

        self.assertGreater(points.shape[0], 0)


    # -----------------------------
    # Test 3: K-Means
    # -----------------------------

    def test_kmeans(self):

        np.random.seed(42)

        points = np.vstack([
            np.random.randn(20, 2) + [0, 0],
            np.random.randn(20, 2) + [10, 10],
            np.random.randn(20, 2) + [20, 20]
        ])

        c1, c2, c3, centroid1, centroid2, centroid3 = k_means(points)

        total_points = len(c1) + len(c2) + len(c3)

        self.assertEqual(
            total_points,
            len(points)
        )


    # -----------------------------
    # Test 4: Centroid Shape
    # -----------------------------

    def test_centroid_shape(self):

        np.random.seed(42)

        points = np.vstack([
            np.random.randn(20, 2) + [0, 0],
            np.random.randn(20, 2) + [10, 10],
            np.random.randn(20, 2) + [20, 20]
        ])

        c1, c2, c3, centroid1, centroid2, centroid3 = k_means(points)

        self.assertEqual(centroid1.shape, (2,))
        self.assertEqual(centroid2.shape, (2,))
        self.assertEqual(centroid3.shape, (2,))


if __name__ == "__main__":
    unittest.main()