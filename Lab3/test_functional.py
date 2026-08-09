from kmeans import (
    load_data,
    choose_columns,
    k_means
)


def functional_test():

    print("Starting Functional Test...\n")

    # Step 1
    df = load_data()

    if df.empty:
        print("FAIL: Dataset is empty")
        return

    print("PASS: Dataset loaded")


    # Step 2
    points = choose_columns(df)

    if points.shape[1] != 2:
        print("FAIL: Incorrect number of features")
        return

    print("PASS: Features extracted")


    # Step 3
    result = k_means(points)

    c1, c2, c3, centroid1, centroid2, centroid3 = result


    # Step 4
    total = len(c1) + len(c2) + len(c3)

    if total != len(points):
        print("FAIL: Some points were lost")
        return

    print("PASS: All points assigned to clusters")


    # Step 5
    if centroid1.shape != (2,):
        print("FAIL: Centroid 1 incorrect")
        return

    if centroid2.shape != (2,):
        print("FAIL: Centroid 2 incorrect")
        return

    if centroid3.shape != (2,):
        print("FAIL: Centroid 3 incorrect")
        return

    print("PASS: Centroids generated")


    print("\nFUNCTIONAL TEST PASSED")


functional_test()